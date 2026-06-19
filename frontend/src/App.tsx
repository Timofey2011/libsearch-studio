import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
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

export default function App() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [collId, setCollId] = useState("");
  const [model, setModel] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SearchResult[]>([]);
  const [busy, setBusy] = useState(false);

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
      const res = await invoke<SearchResult[]>("ask", {
        collectionId: collId,
        question,
        model,
      });
      setSources(res);
    } catch (e) {
      setAnswer("Error: " + String(e));
    }
    setBusy(false);
  }

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 820, margin: "2rem auto", padding: "0 1rem" }}>
      <h1>LibSearch Studio</h1>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <select value={collId} onChange={(e) => setCollId(e.target.value)}>
          {collections.length === 0 && <option value="">(no collections)</option>}
          {collections.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
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
          {answer}
        </div>
      )}
      {sources.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <b>Sources</b>
          <ol>
            {sources.map((s) => (
              <li key={s.rank}>{s.citation}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
