# DOM-side citation harness (ROADMAP-3 §17 follow-up)

Measures TRUE cite-jump outcomes by replaying real store citations through the
app's actual matchers — the vendored foliate-js search for epubs and pdfjs
page text for pdfs — in a browser, closing the proxy gap `ls-cli cite-metric`
documents in its caveats header.

## Run

1. Create a scratch dir with:
   - `index.html` (this file's sibling)
   - `foliate` → symlink to `frontend/src/vendor/foliate`
   - `pdfjs-dist` → symlink to `frontend/node_modules/pdfjs-dist`
   - `pdfjs-assets` → symlink to `frontend/public/pdfjs`
   - `books/` → symlinks to the sampled books
   - `manifest.json` — `{"epub":[{file,title,cases:[{id,chapter,page,cite}]}],"pdf":[…]}`
     (sample with the same FNV-1a/seed scheme as cite-metric so runs align; a
     pylance one-liner over the store works — see §17.2b notes)
2. `python3 -m http.server 8741` in that dir, open `http://localhost:8741/`.
   Add `?only=pdf` or `?only=epub` to run one family — the epub pass takes
   ~15 minutes, so pdf iterations should not have to pay it.
3. The page prints per-book progress and a final `DONE epub={…} pdf={…}` tally
   plus full per-case JSON.

The epub path replicates BookReader.citeJump verbatim (normText/normChapter/
findTocEntry/probe selection, all from `frontend/src/lib/cite.ts`) but drives
the REAL `view.search`; outcomes: direct / located / miss-in-chapter /
cold-miss, plus `chapterResolved` per case — the label-drift diagnostic.

The pdf path replicates `PdfReader.locateCite` byte-for-byte: same item join
(no separator, `\n` only at EOL), same `\s*` probe regex, same page → page∓1 →
4-word ladder. Its outcomes are what the user sees: highlight-on-page /
highlight-near / highlight-short / overlay-miss.

Runs so far:

- 2026-07-24 §17.2c (v0.16.1): epub 25% direct · 60% located (85% land);
  chapter labels resolved in only ~2/12 books — that drift was the epub gap.
- 2026-07-24 §17.2c (v0.16.2): epub **60% direct** · 25% located, same 85%
  land, labels resolving in 85% of books. Threshold frozen: epub ≥80% land.
- 2026-07-24 §17.2d (v0.16.3): pdf **24/24 highlighted** (23 on-page, 1 short
  needle, 0 overlay-miss). Threshold frozen: pdf ≥80% on-page.

CAVEAT: both thresholds are measured on PDFs that HAVE a text layer. A scanned
book has none, so `locateCite` can never match and every citation into it
shows the overlay — see ROADMAP-3 §18.
