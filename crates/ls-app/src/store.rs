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
    ord             INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, ord);
-- Incremental-ingest manifest: book fingerprint per collection.
CREATE TABLE IF NOT EXISTS book_state (
    collection_id TEXT NOT NULL,
    book_id       TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    PRIMARY KEY (collection_id, book_id)
);
"#;

pub struct Db {
    conn: Connection,
}

impl Db {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DbError> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        conn.execute_batch(SCHEMA)?;
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

    pub fn add_message(&self, m: &Message) -> Result<(), DbError> {
        let ord: i64 = self.conn.query_row(
            "SELECT COALESCE(MAX(ord), -1) + 1 FROM messages WHERE conversation_id = ?1",
            params![m.conversation_id],
            |r| r.get(0),
        )?;
        self.conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, citations, ord)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                m.id,
                m.conversation_id,
                m.role.as_str(),
                m.content,
                serde_json::to_string(&m.citations)?,
                ord
            ],
        )?;
        Ok(())
    }

    pub fn list_messages(&self, conversation_id: &str) -> Result<Vec<Message>, DbError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, conversation_id, role, content, citations
             FROM messages WHERE conversation_id = ?1 ORDER BY ord",
        )?;
        let rows = stmt.query_map(params![conversation_id], |r| {
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
            let (id, conversation_id, role, content, cites) = row?;
            let citations: Vec<Citation> = serde_json::from_str(&cites)?;
            out.push(Message {
                id,
                conversation_id,
                role: Role::parse(&role)
                    .ok_or_else(|| DbError::NotFound(format!("bad role {role}")))?,
                content,
                citations,
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
        })
        .unwrap();

        let msgs = db.list_messages("conv1").unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User); // ordering preserved
        assert_eq!(msgs[1].citations.len(), 1);
        assert_eq!(msgs[1].citations[0].page, Some(5));
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
}
