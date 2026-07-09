# Releasing LibSearch Studio

How to cut a version and publish it to GitHub with the packaged `.dmg`.

The publish path is **local**: `scripts/publish_release.sh` builds the `.dmg`
on your Mac and uploads it. Fast, zero CI minutes, and it's the arm64 build you
already test.

The CI path (`.github/workflows/release.yml`) is a **manual fallback only**,
for releasing when away from a Mac: Actions → Release → *Run workflow* → enter
the tag. It no longer fires on tag pushes — macOS runners bill at **10×**, and
the automatic run used to rebuild the locally-published dmg cold (~220 billed
minutes per release, for nothing).

Both produce the same thing: a GitHub Release for tag `vN` with the `.dmg`
attached, install/Gatekeeper notes, and a SHA-256.

---

## One-time: create the GitHub repo + remote

Requires the [GitHub CLI](https://cli.github.com) (`gh auth login` first).

```sh
# From the repo root. Choose --private or --public.
gh repo create Timofey2011/libsearch-studio --private --source=. --remote=origin --push
```

This adds the `origin` remote and pushes `main`. (Substitute your own
owner/name; keep it in sync with the `repository` field in `Cargo.toml`.)

> Nothing sensitive is committed — API keys live in the app's data dir
> (`settings.toml`), never in the repo. Still, skim `git log -p` before making a
> repo **public**.

---

## Cut a release

1. **Bump the four version manifests** to the new `X.Y.Z` (keep them identical):
   - `Cargo.toml` → `[workspace.package] version`
   - `src-tauri/Cargo.toml` → `[package] version`
   - `src-tauri/tauri.conf.json` → `version`
   - `frontend/package.json` → `version`

2. **Build the app + dmg** (from the repo root):

   ```sh
   ./frontend/node_modules/.bin/tauri build --bundles app
   ./scripts/make_dmg.sh
   ```

3. **Verify** the built version matches:

   ```sh
   /usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" \
     "target/release/bundle/macos/LibSearch Studio.app/Contents/Info.plist"
   ```

4. **Commit + tag** (the tag/commit body becomes the release notes):

   ```sh
   git commit -am "Release vX.Y.Z"
   git tag -a vX.Y.Z -m "LibSearch Studio vX.Y.Z"
   ```

5. **Publish** — either:

   ```sh
   scripts/publish_release.sh            # local: pushes + uploads the dmg you built
   ```

   (Pushing the tag alone no longer publishes anything — run the script, or
   manually dispatch the Release workflow from the Actions tab as a fallback.)

   Re-running `publish_release.sh` for an existing tag updates the notes and
   re-uploads the dmg (`--clobber`), so it's safe to re-run.

---

## Notes for people installing a release

The `.dmg` is **not notarized** (no Apple Developer signing), so macOS Gatekeeper
blocks it on first launch. The release notes tell users to **right-click the app
→ Open** (or `xattr -dr com.apple.quarantine "/Applications/LibSearch Studio.app"`).
To remove this friction entirely you'd sign + notarize with an Apple Developer
ID — see Tauri's macOS code-signing guide; that's a separate, paid setup.

Only an Apple-Silicon (`arm64`) `.dmg` is published. Add an Intel build to the
matrix if you need `x86_64`.
