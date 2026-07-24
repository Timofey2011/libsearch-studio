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
3. The page prints per-book progress and a final `DONE epub={…} pdf={…}` tally
   plus full per-case JSON.

The epub path replicates BookReader.citeJump verbatim (normText/normChapter/
probe selection) but drives the REAL `view.search`; outcomes: direct /
located / miss-in-chapter / cold-miss (+ chapterResolved per case — the
fitz-vs-foliate chapter-label drift diagnostic). The pdf path checks the cited
page's pdfjs text for the probe: on-page / near / off-page.

First run (2026-07-24, §17.2b): epub 25% direct · 60% located (85% land),
chapter labels resolving in only ~2/12 books = the drift is the remaining
epub gap; pdf 87.5% on-page. Thresholds frozen on these.
