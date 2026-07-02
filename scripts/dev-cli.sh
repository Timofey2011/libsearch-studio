#!/usr/bin/env bash
# Build + run the ls-cli dev tool against the installed app's real index, using a
# SEPARATE Cargo target dir so it never thrashes the app build cache.
#
# Why the separate target dir: `cargo build -p ls-cli` and the full `tauri build`
# select different workspace subsets. With resolver-2 feature unification, the app
# pulls extra features onto shared low-level crates (tokio, object_store, …), so
# alternating between the two forces the heavy lance/lancedb/datafusion stack to
# recompile each way (~8 min). Pinning the CLI to target-cli/ keeps both caches
# warm and independent — CLI iteration stays ~13s and app builds stay ~1-2 min.
#
#   scripts/dev-cli.sh search "investment for engineers main points"
#   scripts/dev-cli.sh ask "what are the main points of Investing for Programmers"
#
# Everything after the script name is forwarded verbatim to ls-cli.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_DATA="${HOME}/.local/share/libsearch-studio"

# Isolated cache so app builds never invalidate CLI builds (and vice versa).
export CARGO_TARGET_DIR="${ROOT}/target-cli"

# Query the engine fully offline, against the installed app's models + index.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LS_MODELS_DIR="${LS_MODELS_DIR:-${APP_DATA}/models}"
export LS_DB_PATH="${LS_DB_PATH:-${APP_DATA}/lancedb}"

cargo build -p ls-cli --release --manifest-path "${ROOT}/Cargo.toml" >&2
exec "${CARGO_TARGET_DIR}/release/ls-cli" "$@"
