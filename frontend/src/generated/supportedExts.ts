// GENERATED from ls-core by `cargo run -p ls-cli -- gen-exts`. DO NOT EDIT:
// a Rust test asserts this file is byte-identical to the generator's output.
export const KNOWN_EXTS = ["pdf", "epub", "mobi", "azw3", "md", "markdown", "txt", "text", "rst", "adoc", "org", "tex", "ipynb", "html", "htm", "docx", "rtf", "odt", "doc", "fb2", "fb2.zip", "pages", "webarchive", "djvu", "xps"] as const;
export const INGEST_EXTS = ["pdf", "md", "markdown", "txt", "text", "rst", "adoc", "org", "tex", "ipynb", "html", "htm", "epub", "fb2", "fb2.zip", "mobi", "azw3", "xps", "docx", "rtf", "odt", "doc", "pages", "webarchive", "djvu"] as const;
/// Extension -> format family (citation shape / reader-kind routing).
export const EXT_FAMILY: Record<string, string> = {
  "pdf": "pdf",
  "epub": "epub",
  "mobi": "mobi",
  "azw3": "mobi",
  "md": "md",
  "markdown": "md",
  "txt": "txt",
  "text": "txt",
  "rst": "txt",
  "adoc": "txt",
  "org": "txt",
  "tex": "txt",
  "ipynb": "md",
  "html": "html",
  "htm": "html",
  "docx": "docx",
  "rtf": "rtf",
  "odt": "odt",
  "doc": "doc",
  "fb2": "fb2",
  "fb2.zip": "fb2",
  "pages": "pages",
  "webarchive": "webarchive",
  "djvu": "djvu",
  "xps": "xps",
};
/// The one extension-derivation rule: lowercase, LONGEST match against
/// KNOWN_EXTS ("x.fb2.zip" is "fb2.zip", never "zip"). Unknown -> null.
export function extOf(name: string): string | null {
  const lower = (name.split(/[\\/]/).pop() ?? name).toLowerCase();
  let best: string | null = null;
  for (const e of KNOWN_EXTS) {
    if (
      lower.length > e.length + 1 &&
      lower.endsWith(e) &&
      lower[lower.length - e.length - 1] === "."
    ) {
      if (!best || e.length > best.length) best = e;
    }
  }
  return best;
}
