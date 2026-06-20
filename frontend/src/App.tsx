import { useEffect, useRef, useState } from "react";
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

// A cited source, in the shape shared by live results, stored citations, and artifacts.
type Src = { rank: number; citation: string; source_path: string; page: number | null; text?: string };
type ChatMessage = { role: "user" | "assistant"; content: string; sources: Src[] };
type Conversation = { id: string; title: string; collection_ids: string[] };
type BackendMessage = { role: "user" | "assistant"; content: string; citations: Src[] };

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

const toSrc = (r: SearchResult): Src => ({
  rank: r.rank,
  citation: r.citation,
  source_path: r.source_path,
  page: r.page,
  text: r.text,
});

export default function App() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [collId, setCollId] = useState("");
  const [model, setModel] = useState("");
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [reader, setReader] = useState<Reader | null>(null);

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [convId, setConvId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [savedByIdx, setSavedByIdx] = useState<Record<number, string>>({});

  const [managing, setManaging] = useState(false);
  const [newName, setNewName] = useState("");
  const [newPaths, setNewPaths] = useState<string[]>([]);
  const [indexing, setIndexing] = useState(false);
  const [progress, setProgress] = useState<{ n: number; total: number; label: string } | null>(null);
  const [indexNote, setIndexNote] = useState<string | null>(null);

  const currentColl = collections.find((c) => c.id === collId) || null;
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    invoke<Collection[]>("list_collections").then(setCollections).catch(console.error);
    invoke<Conversation[]>("list_conversations").then(setConversations).catch(console.error);
    invoke<string[]>("list_models")
      .then((m) => {
        setModels(m);
        if (m[0]) setModel(m[0]);
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    // Initialize the selection once; don't clobber it when the list grows.
    setCollId((cur) => cur || collections[0]?.id || "");
  }, [collections]);

  // Append streamed tokens to the in-flight assistant message (the last one).
  useEffect(() => {
    const un = listen<string>("ask-token", (e) =>
      setMessages((prev) => {
        if (!prev.length) return prev;
        const last = prev.length - 1;
        if (prev[last].role !== "assistant") return prev;
        const copy = [...prev];
        copy[last] = { ...copy[last], content: copy[last].content + e.payload };
        return copy;
      })
    );
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

  // Keep the transcript scrolled to the latest turn.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  async function send() {
    const q = question.trim();
    if (!collId || !q || busy) return;
    setQuestion("");
    setBusy(true);
    setSavedByIdx({});

    let cid = convId;
    try {
      if (!cid) {
        const c = await invoke<Conversation>("create_conversation", { collectionId: collId, title: q });
        cid = c.id;
        setConvId(c.id);
        setConversations((prev) => [c, ...prev]);
      }
      // Optimistic: show the user turn + an empty assistant turn to stream into.
      setMessages((prev) => [...prev, { role: "user", content: q, sources: [] }, { role: "assistant", content: "", sources: [] }]);
      const res = await invoke<SearchResult[]>("ask", {
        collectionId: collId,
        conversationId: cid,
        question: q,
        model,
      });
      setMessages((prev) => {
        const copy = [...prev];
        const last = copy.length - 1;
        if (copy[last]?.role === "assistant") copy[last] = { ...copy[last], sources: res.map(toSrc) };
        return copy;
      });
    } catch (e) {
      setMessages((prev) => {
        const copy = [...prev];
        const last = copy.length - 1;
        if (copy[last]?.role === "assistant")
          copy[last] = { ...copy[last], content: copy[last].content + `\n[Error: ${String(e)}]` };
        return copy;
      });
    }
    setBusy(false);
  }

  function newChat() {
    setConvId("");
    setMessages([]);
    setSavedByIdx({});
    setReader(null);
  }

  async function openConversation(c: Conversation) {
    setConvId(c.id);
    setSavedByIdx({});
    if (c.collection_ids[0]) setCollId(c.collection_ids[0]);
    const msgs = await invoke<BackendMessage[]>("list_messages", { conversationId: c.id });
    setMessages(msgs.map((m) => ({ role: m.role, content: m.content, sources: m.citations })));
  }

  async function deleteConversation(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    await invoke("delete_conversation", { conversationId: id });
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (convId === id) newChat();
  }

  function openSource(s: Src) {
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
    const coll = await invoke<Collection>("create_collection", { name: newName.trim(), sourcePaths: newPaths });
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

  async function saveArtifact(idx: number) {
    const a = messages[idx];
    const q = messages[idx - 1]?.content ?? "";
    try {
      const path = await invoke<string>("save_artifact", {
        collectionId: collId,
        question: q,
        answer: a.content,
        model,
        created: new Date().toISOString().slice(0, 19).replace("T", " "),
        sources: a.sources,
      });
      setSavedByIdx((prev) => ({ ...prev, [idx]: path }));
    } catch (e) {
      setSavedByIdx((prev) => ({ ...prev, [idx]: "Error: " + String(e) }));
    }
  }

  function chooseModel(m: string) {
    setModel(m);
    invoke("warm_model", { model: m }).catch(console.error);
  }

  // Turn [n] / [n, m] markers into links that jump to the cited page in the reader.
  function renderAnswer(text: string, sources: Src[]) {
    return text.split(/(\[[\d,\s]+\])/g).map((part, i) => {
      const m = part.match(/^\[([\d,\s]+)\]$/);
      if (!m) return <span key={i}>{part}</span>;
      const nums = m[1].split(",").map((s) => s.trim()).filter(Boolean);
      return (
        <span key={i}>
          [
          {nums.map((n, j) => {
            const rank = parseInt(n, 10);
            const s = sources.find((x) => x.rank === rank);
            return (
              <span key={j}>
                {j > 0 ? ", " : ""}
                {s ? (
                  <a
                    onClick={() => openSource(s)}
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

  // WKWebView honors the #page fragment to jump to a page.
  const readerSrc = reader ? convertFileSrc(reader.path) + (reader.page ? `#page=${reader.page}` : "") : "";

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "system-ui, sans-serif" }}>
      {/* Conversation sidebar */}
      <div style={{ width: 210, flexShrink: 0, borderRight: "1px solid #e3e3e3", display: "flex", flexDirection: "column", background: "#fafafa" }}>
        <div style={{ padding: "10px 12px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <b style={{ fontSize: 14 }}>Conversations</b>
          <button onClick={newChat} title="Start a new conversation">+ New</button>
        </div>
        <div style={{ overflow: "auto", flex: 1 }}>
          {conversations.length === 0 && (
            <div style={{ padding: "0 12px", fontSize: 12, color: "#999" }}>No conversations yet.</div>
          )}
          {conversations.map((c) => (
            <div
              key={c.id}
              onClick={() => openConversation(c)}
              style={{
                padding: "8px 12px",
                cursor: "pointer",
                fontSize: 13,
                background: c.id === convId ? "#e7f0ff" : "transparent",
                display: "flex",
                justifyContent: "space-between",
                gap: 6,
                alignItems: "center",
              }}
            >
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.title}</span>
              <button
                onClick={(e) => deleteConversation(c.id, e)}
                title="Delete conversation"
                style={{ border: "none", background: "none", color: "#aaa", cursor: "pointer", padding: 0 }}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Chat column */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <div style={{ display: "flex", gap: 8, padding: "10px 12px", borderBottom: "1px solid #eee", alignItems: "center" }}>
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
          <div style={{ margin: 12, padding: 12, border: "1px solid #ddd", borderRadius: 8, background: "#fafafa" }}>
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

        {/* Transcript */}
        <div ref={scrollRef} style={{ flex: 1, overflow: "auto", padding: "1rem 1.25rem" }}>
          {messages.length === 0 && (
            <div style={{ color: "#999", marginTop: 24 }}>
              Ask a question to start a conversation grounded in your library.
            </div>
          )}
          {messages.map((msg, idx) =>
            msg.role === "user" ? (
              <div key={idx} style={{ margin: "12px 0" }}>
                <div style={{ fontSize: 12, color: "#888", marginBottom: 2 }}>You</div>
                <div style={{ whiteSpace: "pre-wrap" }}>{msg.content}</div>
              </div>
            ) : (
              <div key={idx} style={{ margin: "12px 0 20px" }}>
                <div style={{ whiteSpace: "pre-wrap", padding: 12, background: "#f5f5f5", borderRadius: 8 }}>
                  {msg.content ? renderAnswer(msg.content, msg.sources) : <span style={{ color: "#999" }}>Thinking…</span>}
                </div>
                {msg.sources.length > 0 && (
                  <>
                    <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 10 }}>
                      <button onClick={() => saveArtifact(idx)} title="Save this answer + sources as Markdown">
                        Save as Markdown
                      </button>
                      {savedByIdx[idx] && (
                        <span style={{ fontSize: 12, color: savedByIdx[idx].startsWith("Error") ? "#b00" : "#2a7" }}>
                          {savedByIdx[idx].startsWith("Error") ? savedByIdx[idx] : `Saved → ${savedByIdx[idx]}`}
                        </span>
                      )}
                    </div>
                    <div style={{ marginTop: 10 }}>
                      <b style={{ fontSize: 13 }}>Sources</b>
                      <ol style={{ paddingLeft: 20, margin: "4px 0 0" }}>
                        {msg.sources.map((s) => (
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
                  </>
                )}
              </div>
            )
          )}
        </div>

        {/* Input */}
        <div style={{ borderTop: "1px solid #eee", padding: 12 }}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            rows={2}
            style={{ width: "100%", boxSizing: "border-box", resize: "none" }}
            placeholder="Ask your library…  (Enter to send, Shift+Enter for newline)"
          />
          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 6 }}>
            <button onClick={send} disabled={busy || !collId || !question.trim()}>
              {busy ? "Thinking…" : "Send"}
            </button>
          </div>
        </div>
      </div>

      {/* Reader */}
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
