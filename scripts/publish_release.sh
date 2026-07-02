#!/usr/bin/env bash
# Publish an already-built release to GitHub from your Mac: push the tag, create
# the GitHub Release, and upload the .dmg with install notes + a SHA-256.
#
#   scripts/publish_release.sh          # version from src-tauri/tauri.conf.json
#   scripts/publish_release.sh 0.5.9    # explicit version
#
# Prereqs (does NOT build or tag — do that first):
#   - gh CLI, authenticated:            gh auth login
#   - an 'origin' remote:               see docs/RELEASING.md (first-time setup)
#   - the annotated tag exists:         git tag -a vX -m "LibSearch Studio vX"
#   - the dmg is built:                 tauri build --bundles app && scripts/make_dmg.sh
#
# CI (.github/workflows/release.yml) does the same on `git push origin vX`; this
# script is the local path when you've already built the dmg on your machine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="${1:-$(sed -n 's/.*"version": *"\([0-9][^"]*\)".*/\1/p' src-tauri/tauri.conf.json | head -1)}"
TAG="v${VERSION}"
DMG="target/release/bundle/dmg/LibSearch Studio_${VERSION}_arm64.dmg"

die() { echo "error: $*" >&2; exit 1; }
command -v gh >/dev/null || die "gh (GitHub CLI) not found — https://cli.github.com"
gh auth status >/dev/null 2>&1 || die "not logged in — run 'gh auth login'"
git remote get-url origin >/dev/null 2>&1 || die "no 'origin' remote — create the repo first (see docs/RELEASING.md)"
git rev-parse "$TAG" >/dev/null 2>&1 || die "tag $TAG not found — run: git tag -a $TAG -m \"LibSearch Studio $TAG\""
[ -f "$DMG" ] || die "$DMG not found — build it: tauri build --bundles app && scripts/make_dmg.sh"

# Verify the tag actually points at the built version (guard against a stale dmg).
PLIST="target/release/bundle/macos/LibSearch Studio.app/Contents/Info.plist"
if [ -f "$PLIST" ]; then
  BUILT=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$PLIST" 2>/dev/null || echo "")
  [ "$BUILT" = "$VERSION" ] || die "built app is $BUILT but publishing $VERSION — rebuild before publishing"
fi

echo "→ Pushing main + tags to origin…"
git push origin main --follow-tags

SHA=$(shasum -a 256 "$DMG" | awk '{print $1}')
NOTES="$(mktemp)"
{
  # The tagged commit's body is the human-written release notes.
  git log -1 --format='%b' "$TAG"
  echo
  echo '### Install (macOS, Apple Silicon)'
  echo
  echo 'Open the `.dmg` and drag **LibSearch Studio** to Applications. This build is **not notarized**, so on first launch macOS blocks it — **right-click the app → Open**, then confirm. (Or: `xattr -dr com.apple.quarantine "/Applications/LibSearch Studio.app"`.)'
  echo
  echo '```'
  echo "SHA-256  $SHA"
  echo '```'
} > "$NOTES"

echo "→ Creating release $TAG with $(basename "$DMG")…"
if gh release view "$TAG" >/dev/null 2>&1; then
  gh release upload "$TAG" "$DMG" --clobber
  gh release edit "$TAG" --notes-file "$NOTES"
else
  gh release create "$TAG" "$DMG" --title "LibSearch Studio $TAG" --notes-file "$NOTES"
fi
rm -f "$NOTES"
echo "✓ Published: $(gh release view "$TAG" --json url -q .url)"
