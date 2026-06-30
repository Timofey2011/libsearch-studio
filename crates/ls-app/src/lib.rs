//! Application service layer: settings, collections, conversations (SQLite),
//! and (later) indexing-job orchestration. This is the layer the Tauri bridge
//! calls; it depends on the engine crates but contains no UI code.

pub mod discover;
pub mod service;
pub mod settings;
pub mod store;
pub mod types;

pub use discover::discover_books;
pub use ls_extract::stable_book_id;
pub use service::{
    content_signature, file_fingerprint, IndexEvent, IndexStats, Service, ServiceError,
};
pub use settings::{ProviderCreds, Settings, SettingsError};
pub use store::{Db, DbError};
pub use types::{Citation, Collection, Conversation, Message, Role};

use std::path::PathBuf;

/// Default per-user data directory for the app (DB, settings, collection indexes).
/// Kept off cloud-sync mounts by living under the OS data dir.
pub fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LS_DATA_DIR") {
        return PathBuf::from(dir);
    }
    // XDG-style default; macOS also accepts ~/.local/share. Tauri overrides this
    // with the platform app-data dir when wiring the bridge.
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".local/share/libsearch-studio")
}
