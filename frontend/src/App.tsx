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
type ChatMessage = { role: "user" | "assistant"; content: string; thinking: string; sources: Src[] };
type Conversation = { id: string; title: string; collection_ids: string[] };
type BackendMessage = { role: "user" | "assistant"; content: string; citations: Src[] };

type Reader = { path: string; page: number | null };

// Mirrors ls_app::Settings. Loaded whole and spread on edit so fields this UI
// doesn't surface (e.g. models_dir) are preserved on save.
type ProviderCreds = { api_key: string; model: string };
type Settings = {
  models_dir: string;
  artifacts_dir: string;
  ollama_host: string;
  ollama_model: string;
  llm_provider: string;
  providers: Record<string, ProviderCreds>;
  python_bin: string;
  indexer_script: string;
  hybrid_top_k: number;
  final_top_k: number;
};

const ANTHROPIC_MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-fable-5"];

// Cloud providers (all API-key based; OpenAI-compatible except Anthropic).
const CLOUD_PROVIDERS: { id: string; label: string; keyHint: string; modelHint: string }[] = [
  { id: "anthropic", label: "Anthropic (Claude)", keyHint: "console.anthropic.com", modelHint: "claude-sonnet-4-6" },
  { id: "openai", label: "OpenAI", keyHint: "platform.openai.com/api-keys", modelHint: "gpt-4o" },
  { id: "gemini", label: "Google Gemini", keyHint: "aistudio.google.com/apikey", modelHint: "gemini-2.0-flash" },
  { id: "fireworks", label: "Fireworks AI", keyHint: "fireworks.ai/account/api-keys", modelHint: "accounts/fireworks/models/…" },
  { id: "ollama_cloud", label: "Ollama Cloud", keyHint: "ollama.com/settings/keys", modelHint: "gpt-oss:120b" },
];

// Mirrors ls_app::IndexEvent (serde tag = "kind", snake_case).
type IndexEvent =
  | { kind: "loading" }
  | { kind: "started"; total: number }
  | { kind: "working"; n: number; total: number; path: string }
  | { kind: "embedding"; n: number; total: number; title: string; chunks_done: number; chunks_total: number }
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
  const [collIds, setCollIds] = useState<string[]>([]);
  const [showCollPicker, setShowCollPicker] = useState(false);
  const [model, setModel] = useState("");
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const [histIndex, setHistIndex] = useState<number | null>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const [busy, setBusy] = useState(false);
  const [reader, setReader] = useState<Reader | null>(null);

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [convId, setConvId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [savedByIdx, setSavedByIdx] = useState<Record<number, string>>({});
  const [thinkOpen, setThinkOpen] = useState<Record<number, boolean>>({});
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const [managing, setManaging] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [settingsNote, setSettingsNote] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [newPaths, setNewPaths] = useState<string[]>([]);
  const [indexing, setIndexing] = useState(false);
  const [progress, setProgress] = useState<{ pct: number; label: string } | null>(null);
  const [indexNote, setIndexNote] = useState<string | null>(null);
  const [llmStatus, setLlmStatus] = useState<{ ok: boolean; message: string } | null>(null);

  // Manage operates on the first selected collection.
  const currentColl = collections.find((c) => c.id === collIds[0]) || null;
  const collLabel =
    collIds.length === 0
      ? "Select collections"
      : collIds.length === 1
        ? collections.find((c) => c.id === collIds[0])?.name ?? "1 collection"
        : `${collIds.length} collections`;
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    invoke<Collection[]>("list_collections").then(setCollections).catch(console.error);
    invoke<Conversation[]>("list_conversations").then(setConversations).catch(console.error);
    invoke<Settings>("get_settings")
      .then((s) => {
        setSettings(s);
        refreshModels(s);
      })
      .catch(console.error);
    // Always probe the provider, even if list_models fails (Ollama down) — that
    // is exactly the state the status indicator needs to surface.
    checkLlm("");
  }, []);

  // Populate the model dropdown. Model listing is best-effort: if a cloud
  // provider doesn't expose /models, fall back to the model set in Settings.
  async function refreshModels(s: Settings | null): Promise<string[]> {
    let opts = await invoke<string[]>("list_models").catch(() => [] as string[]);
    const prov = s?.llm_provider;
    if (opts.length === 0 && s && prov && prov !== "ollama") {
      const cm = s.providers[prov]?.model;
      if (cm) opts = [cm];
    }
    setModels(opts);
    setModel((cur) => (opts.includes(cur) ? cur : opts[0] ?? ""));
    return opts;
  }

  useEffect(() => {
    // Default to the first collection once; don't clobber an existing selection.
    setCollIds((cur) => (cur.length ? cur : collections[0] ? [collections[0].id] : []));
  }, [collections]);

  // Append streamed tokens to the in-flight assistant message (the last one).
  useEffect(() => {
    const append = (field: "content" | "thinking") => (e: { payload: string }) =>
      setMessages((prev) => {
        if (!prev.length) return prev;
        const last = prev.length - 1;
        if (prev[last].role !== "assistant") return prev;
        const copy = [...prev];
        copy[last] = { ...copy[last], [field]: copy[last][field] + e.payload };
        return copy;
      });
    const unTok = listen<string>("ask-token", append("content"));
    const unThink = listen<string>("ask-reasoning", append("thinking"));
    return () => {
      unTok.then((f) => f());
      unThink.then((f) => f());
    };
  }, []);

  useEffect(() => {
    const un = listen<IndexEvent>("index-progress", (e) => {
      const ev = e.payload;
      const file = (p: string) => p.split("/").pop();
      if (ev.kind === "loading") setProgress({ pct: 0, label: "Loading models…" });
      else if (ev.kind === "started") setProgress({ pct: 0, label: `Found ${ev.total} file(s)` });
      else if (ev.kind === "working")
        setProgress({ pct: (ev.total ? (ev.n - 1) / ev.total : 0) * 100, label: `Reading ${file(ev.path)} (${ev.n}/${ev.total})` });
      else if (ev.kind === "embedding") {
        const within = ev.chunks_total ? ev.chunks_done / ev.chunks_total : 0;
        const pct = (ev.total ? (ev.n - 1 + within) / ev.total : 0) * 100;
        setProgress({ pct, label: `Indexing ${ev.title} — ${ev.chunks_done}/${ev.chunks_total} chunks (${ev.n}/${ev.total})` });
      } else if (ev.kind === "indexed")
        setProgress({ pct: (ev.n / ev.total) * 100, label: `Indexed ${ev.title}` });
      else if (ev.kind === "unchanged")
        setProgress({ pct: (ev.n / ev.total) * 100, label: `Unchanged ${ev.title}` });
      else if (ev.kind === "skipped")
        setProgress({ pct: (ev.n / ev.total) * 100, label: `Skipped ${file(ev.path)}: ${ev.reason}` });
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

  function toggleColl(id: string) {
    setCollIds((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]));
  }

  async function send() {
    const q = question.trim();
    if (!collIds.length || !q || busy) return;
    setQuestion("");
    setHistory((h) => (h[h.length - 1] === q ? h : [...h, q]));
    setHistIndex(null);
    setBusy(true);
    setSavedByIdx({});

    let cid = convId;
    try {
      if (!cid) {
        const c = await invoke<Conversation>("create_conversation", { collectionIds: collIds, title: q });
        cid = c.id;
        setConvId(c.id);
        setConversations((prev) => [c, ...prev]);
      }
      // Optimistic: show the user turn + an empty assistant turn to stream into.
      setMessages((prev) => [
        ...prev,
        { role: "user", content: q, thinking: "", sources: [] },
        { role: "assistant", content: "", thinking: "", sources: [] },
      ]);
      const res = await invoke<SearchResult[]>("ask", {
        collectionIds: collIds,
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

  // Insert a newline at the caret (Alt/Shift+Enter), keeping the caret after it.
  function insertNewline() {
    const ta = taRef.current;
    if (!ta) return;
    const s = ta.selectionStart;
    const e = ta.selectionEnd;
    setQuestion(question.slice(0, s) + "\n" + question.slice(e));
    requestAnimationFrame(() => {
      if (taRef.current) taRef.current.selectionStart = taRef.current.selectionEnd = s + 1;
    });
  }

  // Shell-style history recall through previously sent questions.
  function recallPrev() {
    if (!history.length) return;
    const idx = histIndex === null ? history.length - 1 : Math.max(0, histIndex - 1);
    setHistIndex(idx);
    setQuestion(history[idx]);
  }
  function recallNext() {
    if (histIndex === null) return;
    const idx = histIndex + 1;
    if (idx >= history.length) {
      setHistIndex(null);
      setQuestion("");
    } else {
      setHistIndex(idx);
      setQuestion(history[idx]);
    }
  }

  function onComposerKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter") {
      if (e.altKey || e.shiftKey) {
        e.preventDefault();
        insertNewline();
      } else {
        e.preventDefault();
        send();
      }
      return;
    }
    const ta = e.currentTarget;
    if (e.key === "ArrowUp" && ta.selectionStart === 0 && history.length) {
      e.preventDefault();
      recallPrev();
    } else if (e.key === "ArrowDown" && ta.selectionStart === ta.value.length && histIndex !== null) {
      e.preventDefault();
      recallNext();
    }
  }

  async function openConversation(c: Conversation) {
    setConvId(c.id);
    setSavedByIdx({});
    if (c.collection_ids.length) setCollIds(c.collection_ids);
    const msgs = await invoke<BackendMessage[]>("list_messages", { conversationId: c.id });
    setMessages(msgs.map((m) => ({ role: m.role, content: m.content, thinking: "", sources: m.citations })));
    setThinkOpen({});
  }

  async function deleteConversation(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    await invoke("delete_conversation", { conversationId: id });
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (convId === id) newChat();
  }

  function startRename(c: Conversation, e: React.MouseEvent) {
    e.stopPropagation();
    setEditingId(c.id);
    setEditTitle(c.title);
  }

  async function commitRename() {
    const id = editingId;
    const title = editTitle.trim();
    setEditingId(null);
    if (!id || !title) return;
    try {
      await invoke("rename_conversation", { conversationId: id, title });
      setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, title } : c)));
    } catch (e) {
      console.error(e);
    }
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
    setCollIds([coll.id]);
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

  async function runFastIndex() {
    if (!currentColl || indexing) return;
    setIndexing(true);
    setIndexNote(null);
    setProgress(null);
    try {
      await invoke<IndexStats>("fast_index_collection", { collectionId: currentColl.id });
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
        collectionIds: collIds,
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

  function checkLlm(m: string) {
    invoke<{ ok: boolean; message: string }>("check_llm", { model: m })
      .then(setLlmStatus)
      .catch((e) => setLlmStatus({ ok: false, message: String(e) }));
  }

  function chooseModel(m: string) {
    setModel(m);
    invoke("warm_model", { model: m }).catch(console.error);
    checkLlm(m);
  }

  function editSetting<K extends keyof Settings>(key: K, value: Settings[K]) {
    setSettings((s) => (s ? { ...s, [key]: value } : s));
  }

  function editCreds(provider: string, field: keyof ProviderCreds, value: string) {
    setSettings((s) => {
      if (!s) return s;
      const cur = s.providers[provider] ?? { api_key: "", model: "" };
      return { ...s, providers: { ...s.providers, [provider]: { ...cur, [field]: value } } };
    });
  }

  async function pickArtifactsDir() {
    const dir = await pickFolder();
    if (dir) editSetting("artifacts_dir", dir);
  }

  async function saveSettings() {
    if (!settings) return;
    try {
      await invoke("save_settings", { settings });
    } catch (e) {
      setSettingsNote("Error: " + String(e));
      return;
    }
    setSettingsNote("Saved.");
    // Refresh the model list (best-effort) and re-check the provider. A failure
    // to list models is not a save failure.
    const opts = await refreshModels(settings);
    checkLlm(opts[0] ?? "");
    setTimeout(() => setSettingsNote(null), 2500);
  }

  // Render a [n] / [n, m] citation marker as links into the reader.
  function renderCitation(tok: string, sources: Src[], key: number) {
    const inner = tok.slice(1, -1);
    const nums = inner.split(",").map((s) => s.trim()).filter(Boolean);
    return (
      <span key={key} className="cite">
        [
        {nums.map((n, j) => {
          const rank = parseInt(n, 10);
          const s = sources.find((x) => x.rank === rank);
          return (
            <span key={j}>
              {j > 0 ? ", " : ""}
              {s ? <a onClick={() => openSource(s)}>{n}</a> : n}
            </span>
          );
        })}
        ]
      </span>
    );
  }

  // Inline markdown: **bold**, *italic*, `code`, and [n] citation links.
  function renderInline(text: string, sources: Src[]) {
    const nodes: React.ReactNode[] = [];
    const re = /\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|\[[\d,\s]+\]/g;
    let last = 0;
    let m: RegExpExecArray | null;
    let i = 0;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) nodes.push(text.slice(last, m.index));
      if (m[1] !== undefined) nodes.push(<strong key={i}>{m[1]}</strong>);
      else if (m[2] !== undefined) nodes.push(<em key={i}>{m[2]}</em>);
      else if (m[3] !== undefined) nodes.push(<code key={i}>{m[3]}</code>);
      else nodes.push(renderCitation(m[0], sources, i));
      last = re.lastIndex;
      i++;
    }
    if (last < text.length) nodes.push(text.slice(last));
    return nodes;
  }

  // Block-level: paragraphs (blank-line separated) and bullet / numbered lists.
  function renderRich(text: string, sources: Src[]) {
    type Block = { type: "p"; text: string } | { type: "ul" | "ol"; items: string[] };
    const blocks: Block[] = [];
    let para: string[] = [];
    let list: { type: "ul" | "ol"; items: string[] } | null = null;
    const flushPara = () => {
      if (para.length) blocks.push({ type: "p", text: para.join(" ") });
      para = [];
    };
    const flushList = () => {
      if (list) blocks.push(list);
      list = null;
    };
    for (const raw of text.split("\n")) {
      const line = raw.trimEnd();
      const bullet = line.match(/^\s*[-*]\s+(.*)$/);
      const numbered = line.match(/^\s*\d+\.\s+(.*)$/);
      if (bullet) {
        flushPara();
        if (!list || list.type !== "ul") {
          flushList();
          list = { type: "ul", items: [] };
        }
        list.items.push(bullet[1]);
      } else if (numbered) {
        flushPara();
        if (!list || list.type !== "ol") {
          flushList();
          list = { type: "ol", items: [] };
        }
        list.items.push(numbered[1]);
      } else if (line.trim() === "") {
        flushPara();
        flushList();
      } else {
        flushList();
        para.push(line);
      }
    }
    flushPara();
    flushList();

    return blocks.map((b, i) => {
      if (b.type === "p") return <p key={i}>{renderInline(b.text, sources)}</p>;
      const items = b.items.map((it, j) => <li key={j}>{renderInline(it, sources)}</li>);
      return b.type === "ul" ? <ul key={i}>{items}</ul> : <ol key={i}>{items}</ol>;
    });
  }

  // WKWebView honors the #page fragment to jump to a page.
  const readerSrc = reader ? convertFileSrc(reader.path) + (reader.page ? `#page=${reader.page}` : "") : "";

  return (
    <div className="app">
      {/* Conversation sidebar */}
      <div className="sidebar">
        <div className="sidebar-head">
          <b>Conversations</b>
          <button className="ghost" onClick={newChat} title="Start a new conversation">
            + New
          </button>
        </div>
        <div className="conv-list">
          {conversations.length === 0 && <div className="muted" style={{ padding: "4px 10px" }}>No conversations yet.</div>}
          {conversations.map((c) =>
            editingId === c.id ? (
              <div key={c.id} className="conv">
                <input
                  autoFocus
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitRename();
                    else if (e.key === "Escape") setEditingId(null);
                  }}
                  style={{ flex: 1, minWidth: 0 }}
                />
              </div>
            ) : (
              <div
                key={c.id}
                className={"conv" + (c.id === convId ? " active" : "")}
                onClick={() => openConversation(c)}
                onDoubleClick={(e) => startRename(c, e)}
                title="Double-click to rename"
              >
                <span className="title">{c.title}</span>
                <button className="ghost del" onClick={(e) => deleteConversation(c.id, e)} title="Delete conversation">
                  ✕
                </button>
              </div>
            )
          )}
        </div>
      </div>

      {/* Chat column */}
      <div className="main">
        <div className="toolbar">
          <div className="coll-picker">
            <button onClick={() => setShowCollPicker((v) => !v)} title="Choose one or more collections to search">
              {collLabel} ▾
            </button>
            {showCollPicker && (
              <div className="coll-menu" onMouseLeave={() => setShowCollPicker(false)}>
                {collections.length === 0 && <div className="muted" style={{ padding: 6 }}>No collections.</div>}
                {collections.map((c) => (
                  <label key={c.id} className="coll-opt">
                    <input type="checkbox" checked={collIds.includes(c.id)} onChange={() => toggleColl(c.id)} />
                    {c.name}
                  </label>
                ))}
              </div>
            )}
          </div>
          <select value={model} onChange={(e) => chooseModel(e.target.value)}>
            {models.length === 0 && <option value="">(no models)</option>}
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          {llmStatus && (
            <span
              className="llm-status"
              title={llmStatus.message}
              onClick={() => checkLlm(model)}
            >
              <span className={"dot " + (llmStatus.ok ? "ok" : "bad")} />
              <span className="muted">{llmStatus.message}</span>
            </span>
          )}
          <span className="spacer" />
          <button onClick={() => setManaging((v) => !v)} title="Add folders and (re)index">
            {managing ? "Done" : "Manage…"}
          </button>
          <button onClick={() => setShowSettings((v) => !v)} title="Settings">
            {showSettings ? "Done" : "Settings"}
          </button>
        </div>

        {showSettings && settings && (
          <div className="panel">
            <h4>Settings</h4>
            <div className="settings-grid">
              <label>Synthesis provider</label>
              <select value={settings.llm_provider} onChange={(e) => editSetting("llm_provider", e.target.value)}>
                <option value="ollama">Ollama (local)</option>
                {CLOUD_PROVIDERS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>

              {settings.llm_provider === "ollama" ? (
                <>
                  <label>Ollama host</label>
                  <input value={settings.ollama_host} onChange={(e) => editSetting("ollama_host", e.target.value)} />
                  <label>Default model</label>
                  <input value={settings.ollama_model} onChange={(e) => editSetting("ollama_model", e.target.value)} />
                </>
              ) : (
                (() => {
                  const p = CLOUD_PROVIDERS.find((x) => x.id === settings.llm_provider)!;
                  const creds = settings.providers[p.id] ?? { api_key: "", model: "" };
                  return (
                    <>
                      <label>API key</label>
                      <input
                        type="password"
                        placeholder={`key from ${p.keyHint}`}
                        value={creds.api_key}
                        onChange={(e) => editCreds(p.id, "api_key", e.target.value)}
                      />
                      <label>Model</label>
                      {p.id === "anthropic" ? (
                        <select value={creds.model || ANTHROPIC_MODELS[1]} onChange={(e) => editCreds(p.id, "model", e.target.value)}>
                          {ANTHROPIC_MODELS.map((m) => (
                            <option key={m} value={m}>
                              {m}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          placeholder={p.modelHint}
                          value={creds.model}
                          onChange={(e) => editCreds(p.id, "model", e.target.value)}
                        />
                      )}
                    </>
                  );
                })()
              )}

              <label>Candidate pool (hybrid_top_k)</label>
              <input
                type="number"
                min={1}
                value={settings.hybrid_top_k}
                onChange={(e) => editSetting("hybrid_top_k", parseInt(e.target.value, 10) || 0)}
              />

              <label>Final results (final_top_k)</label>
              <input
                type="number"
                min={1}
                value={settings.final_top_k}
                onChange={(e) => editSetting("final_top_k", parseInt(e.target.value, 10) || 0)}
              />

              <label>Artifacts folder</label>
              <div className="row">
                <input
                  value={settings.artifacts_dir}
                  onChange={(e) => editSetting("artifacts_dir", e.target.value)}
                  style={{ flex: 1, minWidth: 0 }}
                />
                <button onClick={pickArtifactsDir}>Browse…</button>
              </div>

              <label>Fast index · Python</label>
              <input
                placeholder="/path/to/ebook-kb/.venv/bin/python"
                value={settings.python_bin}
                onChange={(e) => editSetting("python_bin", e.target.value)}
              />
              <label>Fast index · script</label>
              <input
                placeholder="/path/to/scripts/index_to_parquet.py"
                value={settings.indexer_script}
                onChange={(e) => editSetting("indexer_script", e.target.value)}
              />
            </div>
            <div className="row" style={{ marginTop: 10 }}>
              <button className="primary" onClick={saveSettings}>
                Save settings
              </button>
              {settingsNote && (
                <span className={settingsNote.startsWith("Error") ? "note-err" : "note-ok"}>{settingsNote}</span>
              )}
            </div>
            <div className="muted" style={{ marginTop: 6 }}>
              Changes apply to the next question. Cloud API keys are stored locally in plaintext
              (settings.toml) and used only to call that provider. OpenAI, Gemini, Fireworks, and
              Ollama Cloud share one OpenAI-compatible client; after saving, the model dropdown
              lists the provider's models.
            </div>
          </div>
        )}

        {managing && (
          <div className="panel">
            {currentColl && (
              <div style={{ marginBottom: 12 }}>
                <h4>
                  {currentColl.name} — {currentColl.source_paths.length} folder(s)
                </h4>
                {currentColl.source_paths.length > 0 ? (
                  <ul className="path-list">
                    {currentColl.source_paths.map((p) => (
                      <li key={p}>{p}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="muted">No folders yet — add one to index.</div>
                )}
                <div className="row" style={{ marginTop: 6 }}>
                  <button onClick={addFolderToCurrent} disabled={indexing}>
                    Add folder…
                  </button>
                  <button
                    className="primary"
                    onClick={runIndex}
                    disabled={indexing || currentColl.source_paths.length === 0}
                  >
                    {indexing ? "Indexing…" : "Index / Re-index"}
                  </button>
                  {settings?.python_bin && settings?.indexer_script && (
                    <button
                      onClick={runFastIndex}
                      disabled={indexing || currentColl.source_paths.length === 0}
                      title="Embed on the GPU via the Python/MPS helper, then import"
                    >
                      {indexing ? "Indexing…" : "Fast index (GPU)"}
                    </button>
                  )}
                </div>
              </div>
            )}

            {(progress || indexNote) && (
              <div style={{ marginTop: 8 }}>
                {progress && (
                  <>
                    <div className="progress-track">
                      <div className="progress-bar" style={{ width: `${Math.min(100, progress.pct)}%` }} />
                    </div>
                    <div className="muted" style={{ marginTop: 4 }}>
                      {progress.label}
                    </div>
                  </>
                )}
                {indexNote && (
                  <div className={indexNote.startsWith("Error") ? "note-err" : "note-ok"} style={{ marginTop: 4 }}>
                    {indexNote}
                  </div>
                )}
              </div>
            )}

            <div style={{ marginTop: 12, borderTop: "1px solid var(--border)", paddingTop: 10 }}>
              <h4>New collection</h4>
              <div className="row">
                <input
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="Name (e.g. Distributed Systems)"
                  style={{ flex: "1 1 200px", minWidth: 0 }}
                />
                <button onClick={addFolderToNew}>Add folder…</button>
                <button className="primary" onClick={createCollection} disabled={!newName.trim() || newPaths.length === 0}>
                  Create
                </button>
              </div>
              {newPaths.length > 0 && (
                <ul className="path-list" style={{ marginTop: 6 }}>
                  {newPaths.map((p) => (
                    <li key={p}>{p}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}

        {/* Transcript */}
        <div ref={scrollRef} className="transcript">
          {messages.length === 0 && (
            <div className="empty">Ask a question to start a conversation grounded in your library.</div>
          )}
          {messages.map((msg, idx) =>
            msg.role === "user" ? (
              <div key={idx} className="turn user">
                <div className="bubble-user">{msg.content}</div>
              </div>
            ) : (
              <div key={idx} className="turn">
                {msg.thinking && (
                  <div className="thinking">
                    <button
                      className="thinking-toggle"
                      onClick={() => setThinkOpen((o) => ({ ...o, [idx]: !o[idx] }))}
                    >
                      <span className="caret">{thinkOpen[idx] ? "▾" : "▸"}</span> Thinking
                      {!msg.content && <span className="muted"> · reasoning…</span>}
                    </button>
                    {thinkOpen[idx] && <div className="thinking-body">{msg.thinking}</div>}
                  </div>
                )}
                <div className="card-assistant">
                  {msg.content ? (
                    renderRich(msg.content, msg.sources)
                  ) : (
                    <span className="muted">{msg.thinking ? "Reasoning…" : "Thinking…"}</span>
                  )}
                </div>
                {msg.sources.length > 0 && (
                  <>
                    <div className="actions">
                      <button onClick={() => saveArtifact(idx)} title="Save this answer + sources as Markdown">
                        Save as Markdown
                      </button>
                      {savedByIdx[idx] && (
                        <span className={savedByIdx[idx].startsWith("Error") ? "note-err" : "note-ok"}>
                          {savedByIdx[idx].startsWith("Error") ? savedByIdx[idx] : `Saved → ${savedByIdx[idx]}`}
                        </span>
                      )}
                    </div>
                    <div className="sources">
                      <b>Sources</b>
                      <ol>
                        {msg.sources.map((s) => (
                          <li key={s.rank}>
                            <button className="src-link" onClick={() => openSource(s)} title="Open source at the cited page">
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

        {/* Composer */}
        <div className="composer">
          <div className="input-wrap">
            <textarea
              ref={taRef}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={onComposerKeyDown}
              rows={2}
              placeholder="Ask your library…"
            />
            <button
              className="send-icon"
              onClick={send}
              disabled={busy || !collIds.length || !question.trim()}
              title="Send (Enter)"
              aria-label="Send"
            >
              {busy ? (
                <span className="send-spin" />
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="9 10 4 15 9 20" />
                  <path d="M20 4v7a4 4 0 0 1-4 4H4" />
                </svg>
              )}
            </button>
          </div>
          <div className="composer-hint">Enter to send · ⌥Enter for newline · ↑/↓ for history</div>
        </div>
      </div>

      {/* Reader */}
      {reader && (
        <div className="reader">
          <div className="reader-head">
            <span className="name">
              {reader.path.split("/").pop()}
              {reader.page ? ` · p.${reader.page}` : ""}
            </span>
            <button className="ghost" onClick={() => setReader(null)}>
              ✕
            </button>
          </div>
          <iframe key={readerSrc} title="source" src={readerSrc} />
        </div>
      )}
    </div>
  );
}
