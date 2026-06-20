import { useEffect, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

type Collection = {
  id: string;
  name: string;
  db_path: string;
  source_paths: string[];
  embed_model: string;
};

type SearchResult = {
  rank: number;
  score: number;
  citation: string;
  title: string;
  page: number | null;
  source_path: string;
  text: string;
};

type Reader = { path: string; page: number | null };

// Mirrors ls_app::IndexEvent (serde tag = "kind", snake_case).
type IndexEvent =
  | { kind: "started"; total: number }
  | { kind: "indexed"; n: number; total: number; title: string; chunks: number }
  | { kind: "unchanged"; n: number; total: number; title: string }
  | { kind: "skipped"; n: number; total: number; path: string; reason: string }
  | { kind: "finished"; stats: IndexStats };

type IndexStats = {
  books_indexed: number;
  books_unchanged: number;
  books_skipped: number;
  books_failed: number;
  chunks_written: number;
};

export default function App() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [collId, setCollId] = useState("");
  const [model, setModel] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SearchResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [reader, setReader] = useState<Reader | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [asked, setAsked] = useState("");
  const [managing, setManaging] = useState(false);
  const [newName, setNewName] = useState("");
  const [newPaths, setNewPaths] = useState<string[]>([]);
  const [indexing, setIndexing] = useState(false);
  const [progress, setProgress] = useState<{ n: number; total: number; label: string } | null>(null);
  const [indexNote, setIndexNote] = useState<string | null>(null);

  const currentColl = collections.find((c) => c.id === collId) || null;

  useEffect(() => {
    invoke<Collection[]>("list_collections").then(setCollections).catch(console.error);
    invoke<string[]>("list_models")
      .then((m) => {
        setModels(m);
        if (m[0]) setModel(m[0]);
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    // Initialize the selection once; don't clobber it when the list grows
    // (e.g. after creating a new collection, which selects itself).
    setCollId((cur) => cur || collections[0]?.id || "");
  }, [collections]);

  useEffect(() => {
    const un = listen<string>("ask-token", (e) => setAnswer((a) => a + e.payload));
    return () => {
      un.then((f) => f());
    };
  }, []);

  useEffect(() => {
    const un = listen<IndexEvent>("index-progress", (e) => {
      const ev = e.payload;
      if (ev.kind === "started") setProgress({ n: 0, total: ev.total, label: "Starting…" });
      else if (ev.kind === "indexed")
        setProgress({ n: ev.n, total: ev.total, label: `Indexed ${ev.title} (${ev.chunks} chunks)` });
      else if (ev.kind === "unchanged")
        setProgress({ n: ev.n, total: ev.total, label: `Unchanged ${ev.title}` });
      else if (ev.kind === "skipped")
        setProgress({ n: ev.n, total: ev.total, label: `Skipped ${ev.path.split("/").pop()}: ${ev.reason}` });
      else if (ev.kind === "finished") {
        const s = ev.stats;
        setIndexNote(
          `Done — ${s.books_indexed} indexed, ${s.books_unchanged} unchanged, ${s.books_skipped + s.books_failed} skipped, ${s.chunks_written} chunks written.`
        );
      }
    });
    return () => {
      un.then((f) => f());
    };
  }, []);

  async function ask() {
    if (!collId || !question.trim()) return;
    setBusy(true);
    setAnswer("");
    setSources([]);
    setSaved(null);
    setAsked(question.trim());
    try {
      const res = await invoke<SearchResult[]>("ask", { collectionId: collId, question, model });
      setSources(res);
    } catch (e) {
      setAnswer("Error: " + String(e));
    }
    setBusy(false);
  }

  function openSource(s: SearchResult) {
    setReader({ path: s.source_path, page: s.page });
  }

  async function pickFolder(): Promise<string | null> {
    const dir = await open({ directory: true, multiple: false, title: "Choose a folder of PDFs" });
    return typeof dir === "string" ? dir : null;
  }

  async function addFolderToNew() {
    const dir = await pickFolder();
    if (dir && !newPaths.includes(dir)) setNewPaths((p) => [...p, dir]);
  }

  async function createCollection() {
    if (!newName.trim() || newPaths.length === 0) return;
    const coll = await invoke<Collection>("create_collection", {
      name: newName.trim(),
      sourcePaths: newPaths,
    });
    setCollections((cs) => [...cs, coll]);
    setCollId(coll.id);
    setNewName("");
    setNewPaths([]);
    setIndexNote(null);
  }

  async function addFolderToCurrent() {
    if (!currentColl) return;
    const dir = await pickFolder();
    if (!dir || currentColl.source_paths.includes(dir)) return;
    const updated = await invoke<Collection>("set_collection_paths", {
      collectionId: currentColl.id,
      sourcePaths: [...currentColl.source_paths, dir],
    });
    setCollections((cs) => cs.map((c) => (c.id === updated.id ? updated : c)));
  }

  async function runIndex() {
    if (!currentColl || indexing) return;
    setIndexing(true);
    setIndexNote(null);
    setProgress(null);
    try {
      await invoke<IndexStats>("index_collection", { collectionId: currentColl.id });
    } catch (e) {
      setIndexNote("Error: " + String(e));
    }
    setIndexing(false);
    setProgress(null);
  }

  async function saveArtifact() {
    if (!answer || busy) return;
    try {
      const path = await invoke<string>("save_artifact", {
        collectionId: collId,
        question: asked || question,
        answer,
        model,
        created: new Date().toISOString().slice(0, 19).replace("T", " "),
        sources,
      });
      setSaved(path);
    } catch (e) {
      setSaved("Error: " + String(e));
    }
  }

  function chooseModel(m: string) {
    setModel(m);
    // Preload it so the next ask is warm (cold-load otherwise dominates latency).
    invoke("warm_model", { model: m }).catch(console.error);
  }

  function openByRank(rank: number) {
    const s = sources.find((x) => x.rank === rank);
    if (s) openSource(s);
  }

  // Render the answer, turning [n] / [n, m] citation markers into clickable links
  // (active once sources arrive) that jump straight to the cited page.
  function renderAnswer(text: string) {
    return text.split(/(\[[\d,\s]+\])/g).map((part, i) => {
      const m = part.match(/^\[([\d,\s]+)\]$/);
      if (!m) return <span key={i}>{part}</span>;
      const nums = m[1].split(",").map((s) => s.trim()).filter(Boolean);
      return (
        <span key={i}>
          [
          {nums.map((n, j) => {
            const rank = parseInt(n, 10);
            const has = sources.some((x) => x.rank === rank);
            return (
              <span key={j}>
                {j > 0 ? ", " : ""}
                {has ? (
                  <a
                    onClick={() => openByRank(rank)}
                    style={{ color: "#0a58ca", cursor: "pointer", textDecoration: "underline" }}
                  >
                    {n}
                  </a>
                ) : (
                  n
                )}
              </span>
            );
          })}
          ]
        </span>
      );
    });
  }

  // PDF.js / WKWebView honors the #page fragment to jump to a page.
  const readerSrc = reader
    ? convertFileSrc(reader.path) + (reader.page ? `#page=${reader.page}` : "")
    : "";

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "system-ui, sans-serif" }}>
      <div style={{ flex: 1, overflow: "auto", padding: "1rem 1.25rem", minWidth: 0 }}>
        <h1 style={{ marginTop: 0 }}>LibSearch Studio</h1>
        <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
          <select value={collId} onChange={(e) => setCollId(e.target.value)}>
            {collections.length === 0 && <option value="">(no collections)</option>}
            {collections.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
          <select value={model} onChange={(e) => chooseModel(e.target.value)}>
            {models.length === 0 && <option value="">(no models)</option>}
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <button onClick={() => setManaging((v) => !v)} title="Add folders and (re)index">
            {managing ? "Done" : "Manage…"}
          </button>
        </div>

        {managing && (
          <div style={{ marginBottom: 12, padding: 12, border: "1px solid #ddd", borderRadius: 8, background: "#fafafa" }}>
            {currentColl && (
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                  {currentColl.name} — {currentColl.source_paths.length} folder(s)
                </div>
                {currentColl.source_paths.length > 0 ? (
                  <ul style={{ margin: "4px 0", paddingLeft: 18, fontSize: 12, color: "#555" }}>
                    {currentColl.source_paths.map((p) => (
                      <li key={p}>{p}</li>
                    ))}
                  </ul>
                ) : (
                  <div style={{ fontSize: 12, color: "#888" }}>No folders yet — add one to index.</div>
                )}
                <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6 }}>
                  <button onClick={addFolderToCurrent} disabled={indexing}>
                    Add folder…
                  </button>
                  <button onClick={runIndex} disabled={indexing || currentColl.source_paths.length === 0}>
                    {indexing ? "Indexing…" : "Index / Re-index"}
                  </button>
                </div>
              </div>
            )}

            {(progress || indexNote) && (
              <div style={{ marginTop: 8 }}>
                {progress && (
                  <>
                    <div style={{ height: 6, background: "#e3e3e3", borderRadius: 3, overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${progress.total ? (progress.n / progress.total) * 100 : 0}%`,
                          background: "#0a58ca",
                          transition: "width .2s",
                        }}
                      />
                    </div>
                    <div style={{ fontSize: 12, color: "#555", marginTop: 4 }}>
                      {progress.n}/{progress.total} · {progress.label}
                    </div>
                  </>
                )}
                {indexNote && (
                  <div style={{ fontSize: 12, marginTop: 4, color: indexNote.startsWith("Error") ? "#b00" : "#2a7" }}>
                    {indexNote}
                  </div>
                )}
              </div>
            )}

            <div style={{ marginTop: 12, borderTop: "1px solid #e3e3e3", paddingTop: 10 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>New collection</div>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <input
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="Name (e.g. Distributed Systems)"
                  style={{ flex: "1 1 200px", minWidth: 0 }}
                />
                <button onClick={addFolderToNew}>Add folder…</button>
                <button onClick={createCollection} disabled={!newName.trim() || newPaths.length === 0}>
                  Create
                </button>
              </div>
              {newPaths.length > 0 && (
                <ul style={{ margin: "6px 0 0", paddingLeft: 18, fontSize: 12, color: "#555" }}>
                  {newPaths.map((p) => (
                    <li key={p}>{p}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}

        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
          style={{ width: "100%", boxSizing: "border-box" }}
          placeholder="Ask your library…"
        />
        <button onClick={ask} disabled={busy || !collId}>
          {busy ? "Thinking…" : "Ask"}
        </button>

        {answer && (
          <div style={{ whiteSpace: "pre-wrap", marginTop: 16, padding: 12, background: "#f5f5f5", borderRadius: 8 }}>
            {renderAnswer(answer)}
          </div>
        )}

        {answer && !busy && (
          <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 10 }}>
            <button onClick={saveArtifact} title="Save this answer + sources as a Markdown file">
              Save as Markdown
            </button>
            {saved && (
              <span style={{ fontSize: 12, color: saved.startsWith("Error") ? "#b00" : "#2a7" }}>
                {saved.startsWith("Error") ? saved : `Saved → ${saved}`}
              </span>
            )}
          </div>
        )}

        {sources.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <b>Sources</b>
            <ol style={{ paddingLeft: 20 }}>
              {sources.map((s) => (
                <li key={s.rank} style={{ marginBottom: 4 }}>
                  <button
                    onClick={() => openSource(s)}
                    title="Open source at the cited page"
                    style={{
                      background: "none",
                      border: "none",
                      padding: 0,
                      color: "#0a58ca",
                      textAlign: "left",
                      cursor: "pointer",
                      textDecoration: "underline",
                      font: "inherit",
                    }}
                  >
                    {s.citation}
                  </button>
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>

      {reader && (
        <div style={{ flex: 1.2, borderLeft: "1px solid #ddd", display: "flex", flexDirection: "column", minWidth: 0 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 10px", background: "#fafafa", borderBottom: "1px solid #eee" }}>
            <span style={{ fontSize: 13, color: "#555", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {reader.path.split("/").pop()}
              {reader.page ? ` · p.${reader.page}` : ""}
            </span>
            <button onClick={() => setReader(null)}>✕</button>
          </div>
          <iframe key={readerSrc} title="source" src={readerSrc} style={{ flex: 1, border: "none" }} />
        </div>
      )}
    </div>
  );
}
