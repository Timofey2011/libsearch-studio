//! Persistent app settings (TOML).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// API key + model for one cloud provider (keyed by provider id in `Settings`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCreds {
    pub api_key: String,
    pub model: String,
}

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
    /// Active synthesis provider: "ollama" (local) or a cloud id
    /// ("anthropic" | "openai" | "gemini" | "fireworks" | "ollama_cloud").
    pub llm_provider: String,
    /// Per-cloud-provider credentials, keyed by provider id. API keys are stored
    /// in plaintext in settings.toml under the app data dir.
    pub providers: BTreeMap<String, ProviderCreds>,
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
            llm_provider: "ollama".to_string(),
            providers: BTreeMap::new(),
            // Candidates reranked per query. The cross-encoder runs on CPU, so this
            // is the main per-query latency knob; 24 keeps recall high while being
            // ~2x faster than 50. (int8-quantizing the reranker is the next lever.)
            hybrid_top_k: 24,
            final_top_k: 8,
        }
    }
}

impl Settings {
    /// Credentials for a cloud provider (empty if unset).
    pub fn creds(&self, provider: &str) -> ProviderCreds {
        self.providers.get(provider).cloned().unwrap_or_default()
    }

    /// The default synthesis model for the active provider.
    pub fn default_model(&self) -> String {
        match self.llm_provider.as_str() {
            "ollama" => self.ollama_model.clone(),
            "anthropic" => {
                let m = self.creds("anthropic").model;
                if m.is_empty() {
                    "claude-sonnet-4-6".to_string()
                } else {
                    m
                }
            }
            p => self.creds(p).model,
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
        let s = Settings {
            ollama_model: "qwen2.5:7b".into(),
            final_top_k: 8,
            ..Settings::default()
        };
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
        assert_eq!(s.hybrid_top_k, 24); // default preserved
    }
}
