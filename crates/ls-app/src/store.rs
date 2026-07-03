//! SQLite-backed persistence for collections, conversations, and messages.

use std::path::Path;

use rusqlite::{params, Connection};

use crate::types::{Citation, Collection, Conversation, Message, Role};

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("not found: {0}")]
    NotFound(String),
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS collections (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    db_path      TEXT NOT NULL,
    source_paths TEXT NOT NULL,   -- JSON array
    embed_model  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id             TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    collection_ids TEXT NOT NULL, -- JSON array
    created_at     INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    citations       TEXT NOT NULL, -- JSON array
    ord             INTEGER NOT NULL,
    in_tokens       INTEGER NOT NULL DEFAULT 0,
    out_tokens      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, ord);
-- Incremental-ingest manifest: book fingerprint per collection.
CREATE TABLE IF NOT EXISTS book_state (
    collection_id TEXT NOT NULL,
    book_id       TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    content_sig   TEXT NOT NULL DEFAULT '',
    chunker_ver   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (collection_id, book_id)
);
-- User-authored memory ("Ledger, not Brain"): one editable note per scope
-- ('global' or a collection id). The app NEVER writes this autonomously; only
-- explicit user actions land here.
CREATE TABLE IF NOT EXISTS notebook (
    scope      TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
"#;

pub struct Db {
    conn: Connection,
}

/// The chunking scheme version stamped on newly indexed books. Bump when chunk
/// boundaries/metadata improve enough that a re-index is worth recommending
/// (v1 = cross-page paragraph-aware GPU chunking + real loc, v0.5.8). Rows from
/// before the column existed default to 0 = legacy.
pub const CURRENT_CHUNKER_VER: i64 = 1;

/// Run an idempotent `ALTER TABLE … ADD COLUMN` migration: a "duplicate column"
/// error means the migration already ran (fine); anything else — locked DB,
/// disk error, typo — propagates instead of being silently swallowed.
fn alter_ignore_duplicate(conn: &Connection, sql: &str) -> Result<(), DbError> {
    match conn.execute(sql, []) {
        Ok(_) => Ok(()),
        Err(e) if e.to_string().contains("duplicate column name") => Ok(()),
        Err(e) => Err(e.into()),
    }
}

impl Db {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DbError> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        conn.execute_batch(SCHEMA)?;
        // Idempotent column migrations for DBs created by older builds.
        alter_ignore_duplicate(
            &conn,
            "ALTER TABLE messages ADD COLUMN in_tokens INTEGER NOT NULL DEFAULT 0",
        )?;
        alter_ignore_duplicate(
            &conn,
            "ALTER TABLE messages ADD COLUMN out_tokens INTEGER NOT NULL DEFAULT 0",
        )?;
        // Content signature for timestamp-independent dedup (added later).
        alter_ignore_duplicate(
            &conn,
            "ALTER TABLE book_state ADD COLUMN content_sig TEXT NOT NULL DEFAULT ''",
        )?;
        // Chunker version per indexed book; 0 = indexed before the marker existed.
        alter_ignore_duplicate(
            &conn,
            "ALTER TABLE book_state ADD COLUMN chunker_ver INTEGER NOT NULL DEFAULT 0",
        )?;
        Ok(Self { conn })
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self, DbError> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    // ---- collections ----

    pub fn upsert_collection(&self, c: &Collection) -> Result<(), DbError> {
        self.conn.execute(
            "INSERT INTO collections (id, name, db_path, source_paths, embed_model)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, db_path=excluded.db_path,
                source_paths=excluded.source_paths, embed_model=excluded.embed_model",
            params![
                c.id,
                c.name,
                c.db_path,
                serde_json::to_string(&c.source_paths)?,
                c.embed_model
            ],
        )?;
        Ok(())
    }

    pub fn list_collections(&self) -> Result<Vec<Collection>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, db_path, source_paths, embed_model FROM collections ORDER BY name",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, name, db_path, paths_json, embed_model) = row?;
            out.push(Collection {
                id,
                name,
                db_path,
                source_paths: serde_json::from_str(&paths_json)?,
                embed_model,
            });
        }
        Ok(out)
    }

    pub fn delete_collection(&self, id: &str) -> Result<(), DbError> {
        self.conn
            .execute("DELETE FROM collections WHERE id = ?1", params![id])?;
        // Drop the incremental-index fingerprints so re-creating the collection
        // re-indexes cleanly.
        self.conn.execute(
            "DELETE FROM book_state WHERE collection_id = ?1",
            params![id],
        )?;
        // And any collection-scoped notebook entry — no invisible orphans.
        self.conn
            .execute("DELETE FROM notebook WHERE scope = ?1", params![id])?;
        Ok(())
    }

    // ---- notebook (user-authored memory) ----

    /// The user's note for a scope ('global' or a collection id); None if unset.
    pub fn get_note(&self, scope: &str) -> Result<Option<String>, DbError> {
        let mut stmt = self
            .conn
            .prepare("SELECT content FROM notebook WHERE scope = ?1")?;
        let mut rows = stmt.query(params![scope])?;
        Ok(match rows.next()? {
            Some(row) => Some(row.get(0)?),
            None => None,
        })
    }

    /// Upsert the user's note for a scope (explicit user action only).
    pub fn set_note(&self, scope: &str, content: &str) -> Result<(), DbError> {
        self.conn.execute(
            "INSERT INTO notebook (scope, content, updated_at) \
             VALUES (?1, ?2, strftime('%s','now')) \
             ON CONFLICT(scope) DO UPDATE SET content = ?2, updated_at = strftime('%s','now')",
            params![scope, content],
        )?;
        Ok(())
    }

    // ---- conversations ----

    pub fn create_conversation(&self, c: &Conversation) -> Result<(), DbError> {
        self.conn.execute(
            "INSERT INTO conversations (id, title, collection_ids) VALUES (?1, ?2, ?3)",
            params![c.id, c.title, serde_json::to_string(&c.collection_ids)?],
        )?;
        Ok(())
    }

    pub fn list_conversations(&self) -> Result<Vec<Conversation>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, collection_ids FROM conversations ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, title, cids) = row?;
            out.push(Conversation {
                id,
                title,
                collection_ids: serde_json::from_str(&cids)?,
            });
        }
        Ok(out)
    }

    pub fn rename_conversation(&self, id: &str, title: &str) -> Result<(), DbError> {
        self.conn.execute(
            "UPDATE conversations SET title = ?2 WHERE id = ?1",
            params![id, title],
        )?;
        Ok(())
    }

    pub fn delete_conversation(&self, id: &str) -> Result<(), DbError> {
        self.conn
            .execute("DELETE FROM conversations WHERE id = ?1", params![id])?;
        Ok(())
    }

    // ---- messages ----

    /// Delete a single message by id (used by "retry" to drop the old answer).
    pub fn delete_message(&self, id: &str) -> Result<(), DbError> {
        self.conn
            .execute("DELETE FROM messages WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn add_message(&self, m: &Message) -> Result<(), DbError> {
        let ord: i64 = self.conn.query_row(
            "SELECT COALESCE(MAX(ord), -1) + 1 FROM messages WHERE conversation_id = ?1",
            params![m.conversation_id],
            |r| r.get(0),
        )?;
        self.conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, citations, ord, in_tokens, out_tokens)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                m.id,
                m.conversation_id,
                m.role.as_str(),
                m.content,
                serde_json::to_string(&m.citations)?,
                ord,
                m.in_tokens,
                m.out_tokens
            ],
        )?;
        Ok(())
    }

    pub fn list_messages(&self, conversation_id: &str) -> Result<Vec<Message>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, conversation_id, role, content, citations, in_tokens, out_tokens
             FROM messages WHERE conversation_id = ?1 ORDER BY ord",
        )?;
        let rows = stmt.query_map(params![conversation_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, u32>(5)?,
                r.get::<_, u32>(6)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, conversation_id, role, content, cites, in_tokens, out_tokens) = row?;
            let citations: Vec<Citation> = serde_json::from_str(&cites)?;
            out.push(Message {
                id,
                conversation_id,
                role: Role::parse(&role)
                    .ok_or_else(|| DbError::NotFound(format!("bad role {role}")))?,
                content,
                citations,
                in_tokens,
                out_tokens,
            });
        }
        Ok(out)
    }

    // ---- incremental ingest manifest ----

    pub fn book_fingerprint(
        &self,
        collection_id: &str,
        book_id: &str,
    ) -> Result<Option<String>, DbError> {
        let r = self.conn.query_row(
            "SELECT fingerprint FROM book_state WHERE collection_id = ?1 AND book_id = ?2",
            params![collection_id, book_id],
            |r| r.get::<_, String>(0),
        );
        match r {
            Ok(fp) => Ok(Some(fp)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Find an already-indexed book in this collection whose stored fingerprint
    /// matches. Because the fingerprint is path-independent (`size:mtime`), this
    /// recognizes a file that moved to a new path so we can re-point its chunks
    /// instead of re-embedding them.
    pub fn book_id_for_fingerprint(
        &self,
        collection_id: &str,
        fingerprint: &str,
    ) -> Result<Option<String>, DbError> {
        let r = self.conn.query_row(
            "SELECT book_id FROM book_state WHERE collection_id = ?1 AND fingerprint = ?2 LIMIT 1",
            params![collection_id, fingerprint],
            |r| r.get::<_, String>(0),
        );
        match r {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Find an already-indexed book in this collection by content signature
    /// (timestamp-independent). Recognizes the same content even if its mtime
    /// changed or it was duplicated. Empty signatures never match.
    pub fn book_id_for_content(
        &self,
        collection_id: &str,
        content_sig: &str,
    ) -> Result<Option<String>, DbError> {
        if content_sig.is_empty() {
            return Ok(None);
        }
        let r = self.conn.query_row(
            "SELECT book_id FROM book_state WHERE collection_id = ?1 AND content_sig = ?2 LIMIT 1",
            params![collection_id, content_sig],
            |r| r.get::<_, String>(0),
        );
        match r {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Record both the metadata fingerprint and the content signature for a book,
    /// stamped with the CURRENT chunker version (freshly indexed = current).
    pub fn set_book_state(
        &self,
        collection_id: &str,
        book_id: &str,
        fingerprint: &str,
        content_sig: &str,
    ) -> Result<(), DbError> {
        self.set_book_state_ver(
            collection_id,
            book_id,
            fingerprint,
            content_sig,
            CURRENT_CHUNKER_VER,
        )
    }

    /// Like [`Db::set_book_state`] but with an explicit chunker version — used by
    /// backfills recording books whose chunks came from an OLDER scheme (ver 0),
    /// so the re-index nudge stays honest about them.
    pub fn set_book_state_ver(
        &self,
        collection_id: &str,
        book_id: &str,
        fingerprint: &str,
        content_sig: &str,
        chunker_ver: i64,
    ) -> Result<(), DbError> {
        self.conn.execute(
            "INSERT INTO book_state (collection_id, book_id, fingerprint, content_sig, chunker_ver)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(collection_id, book_id)
             DO UPDATE SET fingerprint=excluded.fingerprint, content_sig=excluded.content_sig,
                           chunker_ver=excluded.chunker_ver",
            params![
                collection_id,
                book_id,
                fingerprint,
                content_sig,
                chunker_ver
            ],
        )?;
        Ok(())
    }

    /// Drop one book's fingerprint row (used after re-pointing a moved book to a
    /// new id).
    pub fn delete_book_state(&self, collection_id: &str, book_id: &str) -> Result<(), DbError> {
        self.conn.execute(
            "DELETE FROM book_state WHERE collection_id = ?1 AND book_id = ?2",
            params![collection_id, book_id],
        )?;
        Ok(())
    }

    pub fn set_book_fingerprint(
        &self,
        collection_id: &str,
        book_id: &str,
        fingerprint: &str,
    ) -> Result<(), DbError> {
        self.conn.execute(
            "INSERT INTO book_state (collection_id, book_id, fingerprint) VALUES (?1, ?2, ?3)
             ON CONFLICT(collection_id, book_id) DO UPDATE SET fingerprint=excluded.fingerprint",
            params![collection_id, book_id, fingerprint],
        )?;
        Ok(())
    }

    /// How many of a collection's indexed books were chunked by an OLDER scheme
    /// (before [`CURRENT_CHUNKER_VER`]) — the basis of the re-index nudge.
    pub fn legacy_chunker_count(&self, collection_id: &str) -> Result<usize, DbError> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM book_state WHERE collection_id = ?1 AND chunker_ver < ?2",
            params![collection_id, CURRENT_CHUNKER_VER],
            |r| r.get(0),
        )?;
        Ok(n as usize)
    }

    /// Forget a collection's book fingerprints so the next Index run re-embeds
    /// everything with the current chunker (the explicit "re-chunk" action —
    /// the normal dedup would otherwise skip unchanged books forever).
    pub fn clear_book_state(&self, collection_id: &str) -> Result<usize, DbError> {
        let n = self.conn.execute(
            "DELETE FROM book_state WHERE collection_id = ?1",
            params![collection_id],
        )?;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coll(id: &str) -> Collection {
        Collection {
            id: id.into(),
            name: format!("Area {id}"),
            db_path: format!("/db/{id}"),
            source_paths: vec!["/books/a".into(), "/books/b".into()],
            embed_model: "bge-m3".into(),
        }
    }

    #[test]
    fn collections_crud() {
        let db = Db::open_in_memory().unwrap();
        db.upsert_collection(&coll("c1")).unwrap();
        db.upsert_collection(&coll("c2")).unwrap();
        assert_eq!(db.list_collections().unwrap().len(), 2);

        let mut updated = coll("c1");
        updated.name = "Renamed".into();
        db.upsert_collection(&updated).unwrap();
        let got = db.list_collections().unwrap();
        assert!(got.iter().any(|c| c.id == "c1" && c.name == "Renamed"));
        assert_eq!(got.len(), 2); // upsert, not insert

        db.delete_collection("c1").unwrap();
        assert_eq!(db.list_collections().unwrap().len(), 1);
    }

    #[test]
    fn chunker_ver_tracking_and_reset() {
        let db = Db::open_in_memory().unwrap();
        // Freshly indexed books are stamped current; backfilled imports are legacy.
        db.set_book_state("c1", "fresh", "fp1", "sig1").unwrap();
        db.set_book_state_ver("c1", "imported", "fp2", "sig2", 0)
            .unwrap();
        assert_eq!(db.legacy_chunker_count("c1").unwrap(), 1);
        // Re-indexing the legacy book (normal path) upgrades its stamp.
        db.set_book_state("c1", "imported", "fp2", "sig2").unwrap();
        assert_eq!(db.legacy_chunker_count("c1").unwrap(), 0);
        // Reset forgets fingerprints so a re-index re-embeds everything.
        assert_eq!(db.clear_book_state("c1").unwrap(), 2);
        assert_eq!(db.legacy_chunker_count("c1").unwrap(), 0);
    }

    #[test]
    fn notebook_roundtrip_and_collection_cleanup() {
        let db = Db::open_in_memory().unwrap();
        assert_eq!(db.get_note("global").unwrap(), None);
        db.set_note("global", "Prefers concise answers.").unwrap();
        db.set_note("global", "Prefers concise answers with examples.")
            .unwrap(); // upsert
        assert_eq!(
            db.get_note("global").unwrap().as_deref(),
            Some("Prefers concise answers with examples.")
        );
        // A collection-scoped note dies with its collection; global survives.
        db.upsert_collection(&coll("c1")).unwrap();
        db.set_note("c1", "Finance shelf notes").unwrap();
        db.delete_collection("c1").unwrap();
        assert_eq!(db.get_note("c1").unwrap(), None);
        assert!(db.get_note("global").unwrap().is_some());
    }

    #[test]
    fn conversation_and_messages_roundtrip() {
        let db = Db::open_in_memory().unwrap();
        let conv = Conversation {
            id: "conv1".into(),
            title: "Chat".into(),
            collection_ids: vec!["c1".into()],
        };
        db.create_conversation(&conv).unwrap();

        db.add_message(&Message {
            id: "m1".into(),
            conversation_id: "conv1".into(),
            role: Role::User,
            content: "hello?".into(),
            citations: vec![],
            in_tokens: 0,
            out_tokens: 0,
        })
        .unwrap();
        db.add_message(&Message {
            id: "m2".into(),
            conversation_id: "conv1".into(),
            role: Role::Assistant,
            content: "answer [1]".into(),
            citations: vec![Citation {
                rank: 1,
                citation: "X · p.5".into(),
                source_path: "/b/x.pdf".into(),
                page: Some(5),
                text: "cited".into(),
            }],
            in_tokens: 12,
            out_tokens: 34,
        })
        .unwrap();

        let msgs = db.list_messages("conv1").unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User); // ordering preserved
        assert_eq!(msgs[1].citations.len(), 1);
        assert_eq!(msgs[1].citations[0].page, Some(5));
        assert_eq!((msgs[1].in_tokens, msgs[1].out_tokens), (12, 34));
    }

    #[test]
    fn deleting_conversation_cascades_messages() {
        let db = Db::open_in_memory().unwrap();
        db.create_conversation(&Conversation {
            id: "c".into(),
            title: "t".into(),
            collection_ids: vec![],
        })
        .unwrap();
        db.add_message(&Message {
            id: "m".into(),
            conversation_id: "c".into(),
            role: Role::User,
            content: "hi".into(),
            citations: vec![],
            in_tokens: 0,
            out_tokens: 0,
        })
        .unwrap();
        db.delete_conversation("c").unwrap();
        assert!(db.list_messages("c").unwrap().is_empty());
    }

    #[test]
    fn rename_conversation_updates_title() {
        let db = Db::open_in_memory().unwrap();
        db.create_conversation(&Conversation {
            id: "c".into(),
            title: "old".into(),
            collection_ids: vec![],
        })
        .unwrap();
        db.rename_conversation("c", "new title").unwrap();
        let convs = db.list_conversations().unwrap();
        assert_eq!(convs[0].title, "new title");
    }

    #[test]
    fn book_fingerprint_tracks_changes() {
        let db = Db::open_in_memory().unwrap();
        assert_eq!(db.book_fingerprint("c1", "b1").unwrap(), None);
        db.set_book_fingerprint("c1", "b1", "fp1").unwrap();
        assert_eq!(db.book_fingerprint("c1", "b1").unwrap(), Some("fp1".into()));
        db.set_book_fingerprint("c1", "b1", "fp2").unwrap();
        assert_eq!(db.book_fingerprint("c1", "b1").unwrap(), Some("fp2".into()));
    }

    #[test]
    fn content_sig_finds_moved_book() {
        let db = Db::open_in_memory().unwrap();
        // An empty signature must never match.
        assert_eq!(db.book_id_for_content("c1", "").unwrap(), None);
        // Record a book under its old id with a content signature.
        db.set_book_state("c1", "old", "fp-old", "sig-xyz").unwrap();
        assert_eq!(
            db.book_id_for_content("c1", "sig-xyz").unwrap(),
            Some("old".into())
        );
        // Scoped per collection.
        assert_eq!(db.book_id_for_content("c2", "sig-xyz").unwrap(), None);
        // set_book_fingerprint must preserve the existing content signature.
        db.set_book_fingerprint("c1", "old", "fp-new").unwrap();
        assert_eq!(
            db.book_id_for_content("c1", "sig-xyz").unwrap(),
            Some("old".into())
        );
    }
}
