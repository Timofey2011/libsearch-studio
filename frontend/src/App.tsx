import { useEffect, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

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
    if (collections[0]) setCollId(collections[0].id);
  }, [collections]);

  useEffect(() => {
    const un = listen<string>("ask-token", (e) => setAnswer((a) => a + e.payload));
    return () => {
      un.then((f) => f());
    };
  }, []);

  async function ask() {
    if (!collId || !question.trim()) return;
    setBusy(true);
    setAnswer("");
    setSources([]);
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
        </div>
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
