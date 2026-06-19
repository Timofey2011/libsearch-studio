//! Application-level domain types: collections, conversations, messages.

use serde::{Deserialize, Serialize};

/// A named index "by area of interest" over a set of source folders.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Collection {
    pub id: String,
    pub name: String,
    /// LanceDB directory backing this collection.
    pub db_path: String,
    /// Source folders/files indexed into this collection.
    pub source_paths: Vec<String>,
    /// Embedding model id used to build the index (must match at query time).
    pub embed_model: String,
}

/// Who authored a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            _ => None,
        }
    }
}

/// A citation attached to an assistant message (so the reader/artifacts can use it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    pub source_path: String,
    pub title: String,
    pub page: Option<u32>,
    pub chapter: Option<String>,
    pub loc_start: i64,
    pub text: String,
}

/// A chat message. Assistant turns carry the citations used to ground them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub conversation_id: String,
    pub role: Role,
    pub content: String,
    pub citations: Vec<Citation>,
}

/// A conversation targeting one or more collections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub collection_ids: Vec<String>,
}
