// Copy the pdfjs-dist runtime assets (standard fonts for PDFs without embedded
// fonts, cmaps for CID-keyed text) into public/ so Vite ships them with the app.
// Runs via the predev/prebuild package.json hooks; public/pdfjs is gitignored.
import { cpSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const src = join(root, "node_modules", "pdfjs-dist");
const dst = join(root, "public", "pdfjs");
mkdirSync(dst, { recursive: true });
cpSync(join(src, "standard_fonts"), join(dst, "standard_fonts"), { recursive: true });
cpSync(join(src, "cmaps"), join(dst, "cmaps"), { recursive: true });
// v6 decodes JBIG2/CCITT-fax/JPEG2000 images (scanned books!) in WASM modules.
cpSync(join(src, "wasm"), join(dst, "wasm"), { recursive: true });
console.log("pdfjs assets copied to public/pdfjs");
