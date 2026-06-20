#!/usr/bin/env bash
# Build a plain, installable .dmg from the bundled .app using only hdiutil.
#
# Tauri's own dmg target runs a Finder-prettifying AppleScript that fails in
# non-GUI / sandboxed shells (no Finder automation). This produces a functional
# (un-prettified) drag-to-Applications dmg that works everywhere, including CI.
#
#   cargo tauri build --bundles app      # produce the .app first
#   scripts/make_dmg.sh                   # then wrap it into a .dmg
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/target/release/bundle/macos/LibSearch Studio.app"
OUT_DIR="$ROOT/target/release/bundle/dmg"
VERSION="$(grep -m1 '"version"' "$ROOT/src-tauri/tauri.conf.json" | sed -E 's/.*"version": *"([^"]+)".*/\1/')"
ARCH="$(uname -m)"
DMG="$OUT_DIR/LibSearch Studio_${VERSION}_${ARCH}.dmg"

[ -d "$APP" ] || { echo "missing $APP — run: cargo tauri build --bundles app" >&2; exit 1; }
mkdir -p "$OUT_DIR"
rm -f "$DMG"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

hdiutil create -volname "LibSearch Studio" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
echo "built $DMG"
