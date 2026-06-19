//! Persistent app settings (TOML).

use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Directory holding the ONNX model folders (`bge-m3/`, `bge-reranker-v2-m3/`).
    pub models_dir: String,
    /// Where generated `.md` artifacts are written.
    pub artifacts_dir: String,
    /// Ollama endpoint and default synthesis model.
    pub ollama_host: String,
    pub ollama_model: String,
    /// Retrieval breadth: hybrid candidate pool and final reranked count.
    pub hybrid_top_k: usize,
    pub final_top_k: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            models_dir: "models".to_string(),
            artifacts_dir: "artifacts".to_string(),
            ollama_host: "http://localhost:11434".to_string(),
            ollama_model: "gemma4:12b-mlx".to_string(),
            hybrid_top_k: 50,
            final_top_k: 10,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("serialize: {0}")]
    Serialize(#[from] toml::ser::Error),
}

impl Settings {
    /// Load settings from a TOML file, or return defaults if it does not exist.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SettingsError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SettingsError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_yields_defaults() {
        let s = Settings::load("/nonexistent/abcdef.toml").unwrap();
        assert_eq!(s, Settings::default());
    }

    #[test]
    fn roundtrips_through_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let mut s = Settings::default();
        s.ollama_model = "qwen2.5:7b".into();
        s.final_top_k = 8;
        s.save(&path).unwrap();
        let loaded = Settings::load(&path).unwrap();
        assert_eq!(loaded, s);
    }

    #[test]
    fn partial_toml_fills_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.toml");
        std::fs::write(&path, "ollama_model = \"llama3.1:8b\"\n").unwrap();
        let s = Settings::load(&path).unwrap();
        assert_eq!(s.ollama_model, "llama3.1:8b");
        assert_eq!(s.hybrid_top_k, 50); // default preserved
    }
}
