//! Application service layer: settings, collections, conversations (SQLite),
//! and (later) indexing-job orchestration. This is the layer the Tauri bridge
//! calls; it depends on the engine crates but contains no UI code.

pub mod discover;
pub mod maintenance;
pub mod plan;
pub mod service;
pub mod settings;
pub mod store;
pub mod types;

pub use discover::discover_books;
pub use ls_extract::stable_book_id;
pub use maintenance::{FixOutcome, MaintenanceReport};
pub use plan::{
    plan_index_run, EmbedItem, IndexPlan, PlanCtx, RemapAction, SkipReason, StateRefresh,
};
pub use service::{
    content_signature, file_fingerprint, is_sig_sentinel, IndexEvent, IndexStats, Service,
    ServiceError, CONTENT_SIG_MISSING,
};
pub use settings::{ProviderCreds, Settings, SettingsError};
pub use store::{BookStateHit, Db, DbError};
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

/// If `path` lives under a known cloud-sync mount, return the provider's name.
/// LanceDB and the SQLite DB corrupt when a sync client rewrites their files
/// underneath them, so the app warns when its INDEX/data dir is on one of these
/// (this must NOT be applied to a user's source library folders — reading books
/// from a synced folder is fine). Matches macOS + common cross-platform markers.
pub fn cloud_sync_provider(path: &std::path::Path) -> Option<&'static str> {
    let p = path.to_string_lossy();
    const MARKERS: &[(&str, &str)] = &[
        ("/Library/Mobile Documents", "iCloud Drive"),
        ("/Library/CloudStorage/", "a cloud drive (File Provider)"),
        ("/Dropbox", "Dropbox"),
        ("/.dropbox", "Dropbox"),
        ("/Google Drive", "Google Drive"),
        ("/GoogleDrive", "Google Drive"),
        ("/OneDrive", "OneDrive"),
        ("/pCloud", "pCloud"),
    ];
    MARKERS
        .iter()
        .find(|(marker, _)| p.contains(marker))
        .map(|(_, name)| *name)
}

#[cfg(test)]
mod tests {
    use super::cloud_sync_provider;
    use std::path::Path;

    #[test]
    fn flags_cloud_mounts_not_local_paths() {
        assert_eq!(
            cloud_sync_provider(Path::new("/Users/x/Dropbox/libsearch/lancedb")),
            Some("Dropbox")
        );
        assert_eq!(
            cloud_sync_provider(Path::new(
                "/Users/x/Library/Mobile Documents/com~apple~CloudDocs/lib"
            )),
            Some("iCloud Drive")
        );
        assert_eq!(
            cloud_sync_provider(Path::new("/Users/x/Library/CloudStorage/GoogleDrive-a/lib")),
            Some("a cloud drive (File Provider)")
        );
        assert_eq!(
            cloud_sync_provider(Path::new("/Users/x/.local/share/libsearch-studio/lancedb")),
            None
        );
    }
}
