Vendored copy of foliate-js (MIT), pinned at commit 78914aef4466eb960965702401634c2cb348e9b1
(https://github.com/johnfactotum/foliate-js). Vendored because the API is
explicitly unstable and the app must be fully offline. vendor/zip.js (BSD-3)
and vendor/fflate.js (MIT) are foliate's own vendored copies.

Local modification: the PDF branch of view.js's makeBook throws instead of
importing foliate's pdf.js — PDFs render through the app's own PdfReader,
and foliate's path would drag in a second 13 MB pdfjs build.
