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
type BackendMessage = { role: "user" | "assistant"; content: string; citations: Src[]; in_tokens: number; out_tokens: number };

// mm:ss (or h:mm:ss) from a duration in seconds.
function fmtDur(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(ss)}` : `${m}:${pad(ss)}`;
}

type ToolsTab = "collections" | "synthesis" | "retrieval" | "indexing" | "general" | "help";
const TOOLS_TABS: [ToolsTab, string][] = [
  ["collections", "Collections"],
  ["synthesis", "Synthesis"],
  ["retrieval", "Retrieval"],
  ["indexing", "Indexing"],
  ["general", "General"],
  ["help", "Help"],
];

type SubTheme = { name: string; blurb: string };
type Theme = { name: string; subthemes: SubTheme[] };
type ThemeMap = { generated_at: number; model: string; book_count: number; themes: Theme[] };

// A node in the explorable bubble tree (lazily deepened up to MAX_DEPTH levels).
type BNode = { name: string; blurb?: string; children?: BNode[] };
const MAX_DEPTH = 5;
const BUBBLE_COLORS = [
  "#2563eb", "#7c3aed", "#db2777", "#dc2626", "#ea580c", "#0d9488", "#16a34a",
  "#0891b2", "#4f46e5", "#9333ea", "#c026d3", "#d97706", "#65a30d", "#e11d48",
];

// Weight = number of leaf topics under a node (drives bubble size).
function nodeWeight(n: BNode): number {
  if (!n.children || n.children.length === 0) return 1;
  return n.children.reduce((s, c) => s + nodeWeight(c), 0);
}
function hexToRgba(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}
// A question whose specificity scales with how deep you've drilled.
function levelQuestion(path: string[]): string {
  const trail = path.join(" › ");
  const leaf = path[path.length - 1];
  if (path.length <= 1) return `Give a high-level summary of ${trail || "the library"} as covered in the library.`;
  if (path.length === 2) return `Explain "${leaf}" (within ${path[0]}) — the key ideas and how they connect.`;
  return `Go deep on "${leaf}" in the context of ${path.slice(0, -1).join(" › ")}: specifics, mechanisms, trade-offs, and detailed points.`;
}

// "Ask angles" — preset ways to interrogate a theme to facilitate reasoning.
const ANGLES: { label: string; q: (t: string) => string }[] = [
  { label: "Overview", q: (t) => `Give a clear, well-structured overview of ${t}, drawing on the library.` },
  { label: "Key ideas", q: (t) => `What are the most important concepts and ideas about ${t}? Explain each briefly.` },
  { label: "Compare", q: (t) => `Compare the different perspectives or approaches to ${t} found across the sources.` },
  { label: "Open questions", q: (t) => `What open questions, debates, or unresolved issues surround ${t}?` },
  { label: "Critique", q: (t) => `What are the main criticisms or limitations discussed about ${t}?` },
];

type Reader = { path: string; page: number | null; missing: boolean };

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
  min_relevance: number;
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
  const [tokens, setTokens] = useState({ in: 0, out: 0 });
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const [toolsOpen, setToolsOpen] = useState(false);
  const [toolsTab, setToolsTab] = useState<ToolsTab>("collections");
  const [settings, setSettings] = useState<Settings | null>(null);
  const [settingsNote, setSettingsNote] = useState<string | null>(null);
  // Per-provider key-check result: validated chat models for the dropdown.
  const [probe, setProbe] = useState<
    Record<string, { status: "checking" | "ok" | "err"; message: string; models: string[] }>
  >({});
  const [settingUp, setSettingUp] = useState(false);
  const [setupLog, setSetupLog] = useState<string[]>([]);
  const [newName, setNewName] = useState("");
  const [newPaths, setNewPaths] = useState<string[]>([]);
  const [indexing, setIndexing] = useState(false);
  const [indexKind, setIndexKind] = useState<"cpu" | "gpu" | null>(null);
  const [progress, setProgress] = useState<{ pct: number; label: string } | null>(null);
  const [indexNote, setIndexNote] = useState<string | null>(null);
  const [indexStart, setIndexStart] = useState<number | null>(null);
  const [nowMs, setNowMs] = useState(0);
  const [idxCount, setIdxCount] = useState<{ done: number; total: number; chunks: number }>({ done: 0, total: 0, chunks: 0 });
  const [indexLog, setIndexLog] = useState<string[]>([]);
  const [showIndexLog, setShowIndexLog] = useState(false);
  const [mainTab, setMainTab] = useState<"chat" | "themes">("chat");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [themeMap, setThemeMap] = useState<ThemeMap | null>(null);
  const [buildingMap, setBuildingMap] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  const [openThemes, setOpenThemes] = useState<Record<number, boolean>>({});
  const [themeView, setThemeView] = useState<"explore" | "list">("explore");
  const [exploreTree, setExploreTree] = useState<BNode[]>([]);
  const [focusPath, setFocusPath] = useState<number[]>([]);
  const [deepening, setDeepening] = useState(false);
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
  const logRef = useRef<HTMLPreElement>(null);
  const thinkRef = useRef<HTMLDivElement>(null);

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

  // When the Synthesis tab opens on a cloud provider that already has a key,
  // validate it and populate the model dropdown automatically.
  useEffect(() => {
    if (!toolsOpen || toolsTab !== "synthesis" || !settings) return;
    const prov = settings.llm_provider;
    if (prov === "ollama" || prov === "anthropic") return;
    const key = settings.providers[prov]?.api_key?.trim();
    if (key && !probe[prov]) probeProvider(prov, key);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toolsOpen, toolsTab, settings?.llm_provider]);

  // Populate the model dropdown. Model listing is best-effort: cloud /models can
  // omit chat models and include image/embedding ones (e.g. Fireworks lists flux
  // first — and it returns 401 for a chat request), so we always keep the
  // configured model available and prefer it, and never default to a non-chat one.
  async function refreshModels(s: Settings | null): Promise<string[]> {
    let opts = await invoke<string[]>("list_models").catch(() => [] as string[]);
    const prov = s?.llm_provider;
    const saved = s ? (prov === "ollama" ? s.ollama_model : s.providers[prov ?? ""]?.model) : "";
    if (prov && prov !== "ollama") {
      // Ensure the configured chat model is present (Fireworks etc. may not list it).
      if (saved && !opts.includes(saved)) opts = [saved, ...opts];
      // Drop obvious non-chat models so they can't become the default.
      const nonChat = /flux|stable-diffusion|sdxl|-image|embed|whisper|rerank|clip/i;
      opts = opts.filter((m) => m === saved || !nonChat.test(m));
    }
    if (opts.length === 0 && saved) opts = [saved];
    setModels(opts);
    setModel((cur) => {
      if (saved && opts.includes(saved)) return saved;
      if (opts.includes(cur)) return cur;
      return opts[0] ?? "";
    });
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
    const unUsage = listen<{ in_tokens: number; out_tokens: number }>("ask-usage", (e) =>
      setTokens((t) => ({ in: t.in + e.payload.in_tokens, out: t.out + e.payload.out_tokens }))
    );
    return () => {
      unTok.then((f) => f());
      unThink.then((f) => f());
      unUsage.then((f) => f());
    };
  }, []);

  useEffect(() => {
    const un = listen<IndexEvent>("index-progress", (e) => {
      const ev = e.payload;
      const file = (p: string) => p.split("/").pop();
      if (ev.kind === "loading") setProgress({ pct: 0, label: "Loading models…" });
      else if (ev.kind === "started") {
        setProgress({ pct: 0, label: `Found ${ev.total} file(s)` });
        setIdxCount({ done: 0, total: ev.total, chunks: 0 });
      } else if (ev.kind === "working")
        setProgress({ pct: (ev.total ? (ev.n - 1) / ev.total : 0) * 100, label: `Reading ${file(ev.path)} (${ev.n}/${ev.total})` });
      else if (ev.kind === "embedding") {
        const within = ev.chunks_total ? ev.chunks_done / ev.chunks_total : 0;
        const pct = (ev.total ? (ev.n - 1 + within) / ev.total : 0) * 100;
        setProgress({ pct, label: `Indexing ${ev.title} — ${ev.chunks_done}/${ev.chunks_total} chunks (${ev.n}/${ev.total})` });
      } else if (ev.kind === "indexed") {
        setProgress({ pct: (ev.n / ev.total) * 100, label: `Indexed ${ev.title}` });
        setIdxCount((c) => ({ done: ev.n, total: ev.total, chunks: c.chunks + ev.chunks }));
      } else if (ev.kind === "unchanged") {
        setProgress({ pct: (ev.n / ev.total) * 100, label: `Unchanged ${ev.title}` });
        setIdxCount((c) => ({ ...c, done: ev.n, total: ev.total }));
      } else if (ev.kind === "skipped") {
        setProgress({ pct: (ev.n / ev.total) * 100, label: `Skipped ${file(ev.path)}: ${ev.reason}` });
        setIdxCount((c) => ({ ...c, done: ev.n, total: ev.total }));
      } else if (ev.kind === "finished") {
        const s = ev.stats;
        setIndexNote(
          `Done — ${s.books_indexed} indexed, ${s.books_unchanged} unchanged, ${s.books_skipped + s.books_failed} skipped, ${s.chunks_written} chunks written.`
        );
      }
    });
    const unLog = listen<string>("index-log", (e) =>
      setIndexLog((l) => [...l, e.payload].slice(-300))
    );
    return () => {
      un.then((f) => f());
      unLog.then((f) => f());
    };
  }, []);

  // Tick a clock once a second while indexing so the elapsed timer updates live.
  useEffect(() => {
    if (!indexing) return;
    const t = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(t);
  }, [indexing]);

  // Keep the transcript scrolled to the latest turn.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  // Keep the (fixed-height) indexing log pinned to its latest line.
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [indexLog, showIndexLog]);

  // Keep the streaming chain-of-thought pinned to its newest line.
  useEffect(() => {
    if (thinkRef.current) thinkRef.current.scrollTop = thinkRef.current.scrollHeight;
  }, [messages]);

  // Load any cached theme map for the selected collections.
  useEffect(() => {
    if (!collIds.length) {
      setThemeMap(null);
      return;
    }
    invoke<ThemeMap | null>("get_theme_map", { collectionIds: collIds })
      .then((m) => setThemeMap(m ?? null))
      .catch(() => {});
  }, [collIds]);

  // Rebuild the explorable bubble tree whenever the map changes.
  useEffect(() => {
    if (!themeMap) {
      setExploreTree([]);
      return;
    }
    setExploreTree(
      themeMap.themes.map((t) => ({
        name: t.name,
        children: t.subthemes.map((s) => ({ name: s.name, blurb: s.blurb })),
      }))
    );
    setFocusPath([]);
  }, [themeMap]);

  // Stream setup output from the one-click GPU provisioning.
  useEffect(() => {
    const un = listen<string>("setup-log", (e) => setSetupLog((l) => [...l.slice(-400), e.payload]));
    return () => {
      un.then((f) => f());
    };
  }, []);

  async function runSetup() {
    if (settingUp) return;
    setSettingUp(true);
    setSetupLog(["Starting setup… (this downloads several GB and can take 10–20 min)"]);
    try {
      await invoke("setup_gpu_indexing");
      const s = await invoke<Settings>("get_settings");
      setSettings(s);
    } catch (e) {
      setSetupLog((l) => [...l, "Error: " + String(e)]);
    }
    setSettingUp(false);
  }

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
        retry: false,
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
    setTokens({ in: 0, out: 0 });
  }

  function copyText(text: string, idx: number) {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        setCopiedIdx(idx);
        setTimeout(() => setCopiedIdx((c) => (c === idx ? null : c)), 1200);
      })
      .catch(() => {});
  }

  // Regenerate the answer at message index `idx` (an assistant turn): drop it and
  // re-ask its preceding question in the same conversation (no duplicate turn).
  async function retryFrom(idx: number) {
    if (busy || !convId) return;
    const q = messages[idx - 1]?.content;
    if (!q || messages[idx]?.role !== "assistant") return;
    setBusy(true);
    setSavedByIdx((s) => {
      const c = { ...s };
      delete c[idx];
      return c;
    });
    // Truncate to the question and add a fresh assistant turn to stream into.
    setMessages((prev) => [
      ...prev.slice(0, idx),
      { role: "assistant", content: "", thinking: "", sources: [] },
    ]);
    try {
      const res = await invoke<SearchResult[]>("ask", {
        collectionIds: collIds,
        conversationId: convId,
        question: q,
        model,
        retry: true,
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

  // Start a *fresh* conversation and immediately ask `q` (used by theme launchers).
  // Uses the freshly created id directly to avoid a convId state race.
  async function askNew(q: string) {
    if (!collIds.length || busy) return;
    setMainTab("chat");
    setReader(null);
    setSavedByIdx({});
    setTokens({ in: 0, out: 0 });
    setBusy(true);
    try {
      const c = await invoke<Conversation>("create_conversation", { collectionIds: collIds, title: q });
      setConvId(c.id);
      setConversations((prev) => [c, ...prev]);
      setMessages([
        { role: "user", content: q, thinking: "", sources: [] },
        { role: "assistant", content: "", thinking: "", sources: [] },
      ]);
      const res = await invoke<SearchResult[]>("ask", {
        collectionIds: collIds,
        conversationId: c.id,
        question: q,
        model,
        retry: false,
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

  function askTheme(theme: string, sub: string, angle: (typeof ANGLES)[number]) {
    const subject = sub ? `"${sub}" (within ${theme})` : `"${theme}"`;
    askNew(angle.q(subject));
  }

  async function buildMap() {
    if (!collIds.length || buildingMap) return;
    setBuildingMap(true);
    setMapError(null);
    try {
      const m = await invoke<ThemeMap>("build_theme_map", { collectionIds: collIds, model });
      setThemeMap(m);
      setOpenThemes(Object.fromEntries(m.themes.map((_, i) => [i, true])));
    } catch (e) {
      setMapError(String(e));
    }
    setBuildingMap(false);
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
    setTokens({
      in: msgs.reduce((s, m) => s + (m.in_tokens || 0), 0),
      out: msgs.reduce((s, m) => s + (m.out_tokens || 0), 0),
    });
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

  async function openSource(s: Src) {
    setReader({ path: s.source_path, page: s.page, missing: false });
    // The book may have been moved/renamed since indexing — warn instead of
    // showing a silently-blank reader.
    try {
      const ok = await invoke<boolean>("source_exists", { path: s.source_path });
      setReader((r) => (r && r.path === s.source_path ? { ...r, missing: !ok } : r));
    } catch {
      /* leave as-is; the iframe will render whatever it can */
    }
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

  async function setPaths(coll: Collection, sourcePaths: string[]) {
    const updated = await invoke<Collection>("set_collection_paths", {
      collectionId: coll.id,
      sourcePaths,
    });
    setCollections((cs) => cs.map((c) => (c.id === updated.id ? updated : c)));
  }

  async function addFolderToColl(coll: Collection) {
    const dir = await pickFolder();
    if (!dir || coll.source_paths.includes(dir)) return;
    await setPaths(coll, [...coll.source_paths, dir]);
  }

  async function removeFolderFromColl(coll: Collection, path: string) {
    await setPaths(coll, coll.source_paths.filter((p) => p !== path));
  }

  async function deleteCollectionById(coll: Collection) {
    if (!confirm(`Delete collection "${coll.name}"? This removes its index (not your files).`)) return;
    await invoke("delete_collection", { collectionId: coll.id });
    setCollections((cs) => cs.filter((c) => c.id !== coll.id));
    setCollIds((ids) => ids.filter((id) => id !== coll.id));
  }

  async function chooseProvider(p: string) {
    const next = settings ? { ...settings, llm_provider: p } : null;
    if (next) setSettings(next);
    try {
      await invoke("set_provider", { provider: p });
      const opts = await refreshModels(next);
      checkLlm(opts[0] ?? "");
    } catch (e) {
      console.error(e);
    }
  }

  // Providers usable right now: local Ollama + any cloud provider with a key set
  // (plus the current one, so it always appears).
  function readyProviders(): string[] {
    const ready = ["ollama"];
    if (settings) {
      for (const p of CLOUD_PROVIDERS) {
        if (settings.providers[p.id]?.api_key?.trim()) ready.push(p.id);
      }
      if (!ready.includes(settings.llm_provider)) ready.push(settings.llm_provider);
    }
    return ready;
  }
  const providerLabel = (id: string) =>
    id === "ollama" ? "Ollama" : CLOUD_PROVIDERS.find((p) => p.id === id)?.label ?? id;

  function startIndexUi(kind: "cpu" | "gpu") {
    setIndexing(true);
    setIndexKind(kind);
    setIndexNote(null);
    setProgress(null);
    setIndexLog([]);
    setIdxCount({ done: 0, total: 0, chunks: 0 });
    const t = Date.now();
    setIndexStart(t);
    setNowMs(t);
  }

  async function runIndex() {
    if (!currentColl || indexing) return;
    startIndexUi("cpu");
    try {
      await invoke<IndexStats>("index_collection", { collectionId: currentColl.id });
    } catch (e) {
      setIndexNote("Error: " + String(e));
    }
    setIndexing(false);
    setIndexKind(null);
    setIndexStart(null);
    setProgress(null);
  }

  // GPU helper is "ready" once Settings has a Python interpreter + indexer script.
  const gpuReady = !!(settings?.python_bin?.trim() && settings?.indexer_script?.trim());

  // One Index button: use the GPU helper when it's set up, else the CPU engine.
  function runAutoIndex() {
    if (gpuReady) runFastIndex();
    else runIndex();
  }

  async function stopIndex() {
    try {
      await invoke("cancel_indexing");
      setIndexNote("Stopping…");
    } catch (e) {
      setIndexNote("Error: " + String(e));
    }
  }

  async function runFastIndex() {
    if (!currentColl || indexing) return;
    startIndexUi("gpu");
    try {
      await invoke<IndexStats>("fast_index_collection", { collectionId: currentColl.id });
    } catch (e) {
      setIndexNote("Error: " + String(e));
    }
    setIndexing(false);
    setIndexKind(null);
    setIndexStart(null);
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
    // Persist the choice so it sticks across relaunches (per provider).
    const s = settings;
    if (!s || !m) return;
    const prov = s.llm_provider;
    const next: Settings =
      prov === "ollama"
        ? { ...s, ollama_model: m }
        : { ...s, providers: { ...s.providers, [prov]: { ...(s.providers[prov] ?? { api_key: "", model: "" }), model: m } } };
    setSettings(next);
    invoke("save_settings", { settings: next }).catch(console.error);
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
    // Editing the key invalidates a prior check for this provider.
    if (field === "api_key")
      setProbe((p) => {
        const { [provider]: _drop, ...rest } = p;
        return rest;
      });
  }

  // Validate a provider's key and fetch its chat models for the dropdown, without
  // saving. Auto-picks the first model if none is chosen yet.
  async function probeProvider(provider: string, key: string) {
    setProbe((p) => ({ ...p, [provider]: { status: "checking", message: "Checking…", models: [] } }));
    try {
      const r = await invoke<{ ok: boolean; message: string; models: string[] }>("probe_provider", {
        provider,
        apiKey: key,
      });
      setProbe((p) => ({
        ...p,
        [provider]: { status: r.ok ? "ok" : "err", message: r.message, models: r.models },
      }));
      const cur = settings?.providers[provider]?.model ?? "";
      if (r.ok && r.models.length > 0 && !r.models.includes(cur)) {
        editCreds(provider, "model", r.models[0]);
      }
    } catch (e) {
      setProbe((p) => ({ ...p, [provider]: { status: "err", message: String(e), models: [] } }));
    }
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
    setSettingsNote(null);
    setToolsOpen(false);
    // Refresh the model list (best-effort) and re-check the provider. A failure
    // to list models is not a save failure.
    const opts = await refreshModels(settings);
    checkLlm(opts[0] ?? "");
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

  // ---- Tools modal tabs (rendered as plain calls so inputs keep focus) ----

  function renderCollectionsTab() {
    return (
      <div>
        {currentColl && (
          <div style={{ marginBottom: 12 }}>
            <h4>
              {currentColl.name} — {currentColl.source_paths.length} folder(s)
            </h4>
            {currentColl.source_paths.length > 0 ? (
              <ul className="path-list">
                {currentColl.source_paths.map((p) => (
                  <li key={p}>
                    <span className="path">{p}</span>
                    <button
                      className="ghost folder-x"
                      onClick={() => removeFolderFromColl(currentColl, p)}
                      disabled={indexing}
                      title="Remove this folder"
                    >
                      Remove
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="muted">No folders yet — add one to index.</div>
            )}
            <div className="row" style={{ marginTop: 6 }}>
              <button onClick={() => addFolderToColl(currentColl)} disabled={indexing}>
                Add folder…
              </button>
              <button
                className="primary"
                onClick={runAutoIndex}
                disabled={indexing || currentColl.source_paths.length === 0}
                title={gpuReady ? "Embed on the GPU (resumable)" : "Embed on the CPU — set up GPU in Settings → Indexing for ~10x faster"}
              >
                {indexing ? "Indexing…" : "Index / Re-index"}
              </button>
              {!indexing &&
                (gpuReady ? (
                  <span className="muted" style={{ fontSize: 11.5 }} title="Using the GPU helper">
                    GPU
                  </span>
                ) : (
                  <span className="muted" style={{ fontSize: 11.5 }}>
                    CPU · <a onClick={() => setToolsTab("indexing")} style={{ cursor: "pointer" }}>set up GPU</a> for ~10× faster
                  </span>
                ))}
              {indexing && (
                <button className="stop-btn" onClick={stopIndex} title="Stop indexing (keeps books already indexed)">
                  ■ Stop
                </button>
              )}
              <span className="spacer" />
              <button onClick={() => deleteCollectionById(currentColl)} disabled={indexing} title="Delete this collection">
                Delete collection
              </button>
            </div>
          </div>
        )}

        {(progress || indexNote || indexLog.length > 0) && (
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
            {indexing && indexStart && (() => {
              const elapsed = Math.max(0, (nowMs - indexStart) / 1000);
              const rate = elapsed > 0 ? idxCount.chunks / elapsed : 0;
              const eta =
                idxCount.done > 0 && idxCount.total > 0
                  ? (elapsed / idxCount.done) * (idxCount.total - idxCount.done)
                  : null;
              return (
                <div className="muted idx-meta" style={{ marginTop: 4 }}>
                  Elapsed {fmtDur(elapsed)}
                  {idxCount.total > 0 && ` · ${idxCount.done}/${idxCount.total} books`}
                  {idxCount.chunks > 0 && ` · ${idxCount.chunks.toLocaleString()} chunks`}
                  {rate > 0 && ` · ${rate.toFixed(0)} ch/s`}
                  {eta != null && ` · ETA ${fmtDur(eta)}`}
                </div>
              );
            })()}
            {indexLog.length > 0 && (
              <div style={{ marginTop: 6 }}>
                <button className="ghost" onClick={() => setShowIndexLog((v) => !v)}>
                  {showIndexLog ? "▾" : "▸"} Log ({indexLog.length})
                </button>
                {showIndexLog && (
                  <pre ref={logRef} className="setup-log index-log">
                    {indexLog.slice(-200).join("\n")}
                  </pre>
                )}
              </div>
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
    );
  }

  function renderSynthesisTab() {
    if (!settings) return null;
    return (
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
            const pr = probe[p.id];
            const isAnthropic = p.id === "anthropic";
            // Dropdown options = validated chat models, plus the currently-saved
            // model even if the provider's /models omitted it (Fireworks does).
            const opts = isAnthropic ? ANTHROPIC_MODELS : pr?.models ?? [];
            const listed = creds.model && !opts.includes(creds.model) ? [creds.model, ...opts] : opts;
            const useDropdown = listed.length > 0;
            return (
              <>
                <label>API key</label>
                <div className="key-row">
                  <input
                    type="password"
                    placeholder={`key from ${p.keyHint}`}
                    value={creds.api_key}
                    onChange={(e) => editCreds(p.id, "api_key", e.target.value)}
                  />
                  <button
                    className="mini"
                    disabled={!creds.api_key.trim() || pr?.status === "checking"}
                    onClick={() => probeProvider(p.id, creds.api_key)}
                  >
                    {pr?.status === "checking" ? "Checking…" : "Check key"}
                  </button>
                </div>
                {pr && (
                  <>
                    <span />
                    <div className={`probe-note ${pr.status}`}>
                      {pr.status === "ok" ? "✓ " : pr.status === "err" ? "✕ " : ""}
                      {pr.message}
                    </div>
                  </>
                )}
                <label>Model</label>
                {useDropdown ? (
                  <select value={creds.model} onChange={(e) => editCreds(p.id, "model", e.target.value)}>
                    {listed.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    placeholder={pr?.status === "checking" ? "checking key…" : p.modelHint}
                    value={creds.model}
                    onChange={(e) => editCreds(p.id, "model", e.target.value)}
                  />
                )}
                {/* Manual override for OpenAI-compat providers whose /models omits
                    a valid chat model (e.g. some Fireworks models). */}
                {!isAnthropic && useDropdown && (
                  <>
                    <label className="sub">or model id</label>
                    <input
                      placeholder={p.modelHint}
                      value={creds.model}
                      onChange={(e) => editCreds(p.id, "model", e.target.value)}
                    />
                  </>
                )}
              </>
            );
          })()
        )}
        <div className="tools-note muted">
          Cloud API keys are stored locally in plaintext (settings.toml) and used only to call that provider. Click{" "}
          <b>Check key</b> to validate it and load that provider's chat models into the dropdown (image, embedding, and
          audio models are filtered out). If a valid chat model isn't listed, type its id in the “or model id” field.
        </div>
      </div>
    );
  }

  function renderRetrievalTab() {
    if (!settings) return null;
    return (
      <div className="settings-grid">
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
        <label>Min relevance (0–1)</label>
        <input
          type="number"
          min={0}
          max={1}
          step={0.05}
          value={settings.min_relevance}
          onChange={(e) => editSetting("min_relevance", parseFloat(e.target.value) || 0)}
        />
        <div className="tools-note muted">
          Min relevance drops weak passages from the sources; a query with no passage above it answers "no matching
          passages" with no sources. Raise to be stricter.
        </div>
      </div>
    );
  }

  function renderIndexingTab() {
    if (!settings) return null;
    return (
      <div>
        <div className="settings-grid">
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
        <div style={{ marginTop: 10 }}>
          <button onClick={runSetup} disabled={settingUp} title="Create a local venv, install deps, and download/export the models">
            {settingUp ? "Setting up…" : "Set up GPU indexing (auto)"}
          </button>
          <div className="muted" style={{ marginTop: 6 }}>
            One-click: local venv + models. Downloads several GB; restart after it finishes.
          </div>
          {setupLog.length > 0 && <pre className="setup-log">{setupLog.join("\n")}</pre>}
        </div>
      </div>
    );
  }

  function renderGeneralTab() {
    if (!settings) return null;
    return (
      <div className="settings-grid">
        <label>Artifacts folder</label>
        <div className="row">
          <input
            value={settings.artifacts_dir}
            onChange={(e) => editSetting("artifacts_dir", e.target.value)}
            style={{ flex: 1, minWidth: 0 }}
          />
          <button onClick={pickArtifactsDir}>Browse…</button>
        </div>
        <label>Models folder</label>
        <input value={settings.models_dir} onChange={(e) => editSetting("models_dir", e.target.value)} />
      </div>
    );
  }

  function renderHelpTab() {
    const stages: { n: number; t: [string, string]; solid: boolean }[] = [
      { n: 1, t: ["Your", "question"], solid: false },
      { n: 2, t: ["Search your", "library"], solid: false },
      { n: 3, t: ["Re-rank", "best matches"], solid: false },
      { n: 4, t: ["LLM reads", "+ writes"], solid: true },
      { n: 5, t: ["Answer +", "citations"], solid: false },
    ];
    const BW = 96, BH = 54, GAP = 18, Y = 60;
    const xs = stages.map((_, i) => 6 + i * (BW + GAP));
    const cx = (i: number) => xs[i] + BW / 2;
    return (
      <div className="help">
        <h3>How LibSearch answers from your library</h3>
        <p>
          A large language model (LLM) is great at writing, but on its own it has never read
          <i> your</i> books, it can't cite where an answer came from, and it will confidently make
          things up. LibSearch fixes that with <b>RAG — Retrieval-Augmented Generation</b>: before the
          model answers, it finds the most relevant passages in your own library and hands them over
          as evidence. The model then answers <i>only</i> from those passages and cites them.
        </p>

        <div className="help-cols">
          <div className="help-card bad">
            <div className="hc-title">✗ Plain LLM</div>
            <ul>
              <li>Doesn't know your books</li>
              <li>Can invent facts (hallucinate)</li>
              <li>No sources to verify</li>
              <li>Frozen at its training cutoff</li>
            </ul>
          </div>
          <div className="help-card good">
            <div className="hc-title">✓ LibSearch (RAG)</div>
            <ul>
              <li>Answers from your library</li>
              <li>Grounded in real passages</li>
              <li>Clickable page-level citations</li>
              <li>Re-index to add new books anytime</li>
            </ul>
          </div>
        </div>

        <h4>What happens when you ask</h4>
        <svg viewBox="0 0 580 230" className="help-diagram" role="img" aria-label="RAG pipeline diagram">
          <defs>
            <marker id="ha" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="var(--muted)" />
            </marker>
          </defs>
          {xs.slice(0, -1).map((x, i) => (
            <line key={i} x1={x + BW} y1={Y + BH / 2} x2={xs[i + 1] - 2} y2={Y + BH / 2}
              stroke="var(--muted)" strokeWidth="1.5" markerEnd="url(#ha)" />
          ))}
          {/* library feeding step 2 */}
          <g>
            <ellipse cx={cx(1)} cy={158} rx={BW / 2} ry="8" fill="var(--panel-2)" stroke="var(--border)" />
            <rect x={xs[1]} y={158} width={BW} height="34" fill="var(--panel-2)" stroke="var(--border)" />
            <ellipse cx={cx(1)} cy={192} rx={BW / 2} ry="8" fill="var(--panel-2)" stroke="var(--border)" />
            <text x={cx(1)} y={181} textAnchor="middle" fontSize="10.5" fill="var(--muted)">your books</text>
            <line x1={cx(1)} y1={156} x2={cx(1)} y2={Y + BH + 2} stroke="var(--muted)" strokeWidth="1.5" markerEnd="url(#ha)" />
          </g>
          {stages.map((s, i) => (
            <g key={s.n}>
              <rect x={xs[i]} y={Y} width={BW} height={BH} rx="9"
                fill={s.solid ? "var(--accent)" : "var(--accent-soft)"} stroke="var(--accent)" />
              <circle cx={xs[i] + 13} cy={Y + 13} r="8" fill={s.solid ? "#fff" : "var(--accent)"} />
              <text x={xs[i] + 13} y={Y + 16.5} textAnchor="middle" fontSize="10" fontWeight="700"
                fill={s.solid ? "var(--accent)" : "#fff"}>{s.n}</text>
              <text x={cx(i)} y={Y + BH / 2 - 1} textAnchor="middle" fontSize="11"
                fill={s.solid ? "#fff" : "var(--text)"}>
                <tspan x={cx(i)} dy="0">{s.t[0]}</tspan>
                <tspan x={cx(i)} dy="13">{s.t[1]}</tspan>
              </text>
            </g>
          ))}
        </svg>

        <ol className="help-steps">
          <li><b>Your question</b> is turned into a numeric “embedding” — a point in meaning-space.</li>
          <li><b>Search your library</b> finds passages nearby in meaning <i>and</i> by keyword (hybrid search), so both “what you meant” and “the exact words” match.</li>
          <li><b>Re-rank</b> scores those candidates with a precise cross-encoder and keeps the best few.</li>
          <li><b>The LLM</b> (local Ollama or a cloud provider you choose) gets your question <i>plus</i> those passages and writes an answer using only them.</li>
          <li><b>The answer</b> comes back with <span className="cite">[1]</span> markers; click one to open the source PDF at the exact page.</li>
        </ol>

        <h4>Why the results are better &amp; more tuned</h4>
        <ul className="help-why">
          <li><b>Grounded</b> — answers come from your sources, so far less hallucination.</li>
          <li><b>Verifiable</b> — every claim links to the page it came from.</li>
          <li><b>Private</b> — retrieval and embeddings run locally; only the final wording is sent to your chosen LLM (and with Ollama, nothing leaves your machine).</li>
          <li><b>Yours</b> — it reasons over <i>your</i> library, including books no public model has seen.</li>
        </ul>

        <h4>Getting started</h4>
        <ol className="help-steps">
          <li><b>Collections</b> tab → add the folder(s) of PDFs you want to search, then <b>Index</b>.</li>
          <li><b>Synthesis</b> tab → pick a provider (local <i>Ollama</i> needs no key; cloud providers need an API key).</li>
          <li>Pick which library to search in the bar under the chat box, then ask away — and click the citations.</li>
        </ol>
      </div>
    );
  }

  // ---- Explore (bubble) view helpers ----
  function nodeAt(path: number[]): BNode | null {
    let nodes = exploreTree;
    let node: BNode | null = null;
    for (const i of path) {
      node = nodes[i] ?? null;
      if (!node) return null;
      nodes = node.children ?? [];
    }
    return node;
  }
  function focusNames(): string[] {
    let nodes = exploreTree;
    const names: string[] = [];
    for (const i of focusPath) {
      const n = nodes[i];
      if (!n) break;
      names.push(n.name);
      nodes = n.children ?? [];
    }
    return names;
  }
  async function deepenFocus() {
    const names = focusNames();
    if (!names.length || deepening) return;
    setDeepening(true);
    setMapError(null);
    try {
      const subs = await invoke<SubTheme[]>("deepen_theme", { model, path: names });
      setExploreTree((prev) => {
        const copy: BNode[] = JSON.parse(JSON.stringify(prev));
        let nodes = copy;
        let n: BNode | null = null;
        for (const i of focusPath) {
          n = nodes[i];
          if (!n) return prev;
          if (!n.children) n.children = [];
          nodes = n.children;
        }
        if (n) n.children = subs.map((s) => ({ name: s.name, blurb: s.blurb }));
        return copy;
      });
    } catch (e) {
      setMapError(String(e));
    }
    setDeepening(false);
  }

  function renderExplore() {
    const focus = focusPath.length ? nodeAt(focusPath) : null;
    const children = focus ? focus.children ?? [] : exploreTree;
    const names = focusNames();
    const colorIdx = focusPath.length ? focusPath[0] : -1;
    const maxW = Math.max(1, ...children.map(nodeWeight));
    return (
      <div className="explore">
        <div className="crumbs">
          <button className={focusPath.length ? "" : "here"} onClick={() => setFocusPath([])}>Library</button>
          {names.map((nm, i) => (
            <span key={i}>
              <span className="sep">›</span>
              <button className={i === names.length - 1 ? "here" : ""} onClick={() => setFocusPath(focusPath.slice(0, i + 1))}>
                {nm}
              </button>
            </span>
          ))}
        </div>

        {focus && (
          <div className="focus-panel">
            {focus.blurb && <div className="muted focus-blurb">{focus.blurb}</div>}
            <div className="ask-row">
              <div className="ask-q">{levelQuestion(names)}</div>
              <button className="primary" disabled={busy} onClick={() => askNew(levelQuestion(names))}>
                Ask ↵
              </button>
            </div>
            <div className="angles">
              {ANGLES.map((a) => (
                <button
                  key={a.label}
                  className="angle"
                  disabled={busy}
                  onClick={() => askNew(a.q(`"${focus.name}" (${names.slice(0, -1).join(" › ") || "the library"})`))}
                >
                  {a.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {children.length > 0 ? (
          <div className="bubbles">
            {children.map((c, i) => {
              const w = nodeWeight(c);
              const d = Math.round(74 + 76 * Math.sqrt(w / maxW));
              const base = BUBBLE_COLORS[(colorIdx >= 0 ? colorIdx : i) % BUBBLE_COLORS.length];
              const bg = colorIdx >= 0 ? hexToRgba(base, 0.5 + 0.12 * (i % 4)) : base;
              return (
                <button
                  key={i}
                  className="bubble"
                  style={{ width: d, height: d, background: bg, fontSize: Math.max(11, Math.min(15, d / 7)) }}
                  title={c.blurb || c.name}
                  onClick={() => setFocusPath([...focusPath, i])}
                >
                  <span className="b-name">{c.name}</span>
                  {w > 1 && <span className="b-count">{w}</span>}
                </button>
              );
            })}
          </div>
        ) : (
          focus && (
            <div className="deepen-box">
              {names.length < MAX_DEPTH ? (
                <button onClick={deepenFocus} disabled={deepening}>
                  {deepening ? "Deepening…" : "Deepen with AI ↓"}
                </button>
              ) : (
                <span className="muted">Deepest level reached.</span>
              )}
              <span className="muted" style={{ marginLeft: 10, fontSize: 11.5 }}>
                Break this into finer sub-topics (five-whys), then ask more specific questions.
              </span>
            </div>
          )
        )}
      </div>
    );
  }

  function renderThemesView() {
    return (
      <div className="themes">
        <div className="themes-head">
          <div>
            <h2>Library map</h2>
            {themeMap ? (
              <div className="muted">
                {themeMap.book_count} books · generated {new Date(themeMap.generated_at).toLocaleString()} · {themeMap.model}
              </div>
            ) : (
              <div className="muted">A map of {collLabel} into themes you can explore — click an angle to launch a focused, grounded question.</div>
            )}
          </div>
          <div className="row">
            {themeMap && !buildingMap && (
              <div className="seg">
                <button className={themeView === "explore" ? "on" : ""} onClick={() => setThemeView("explore")}>Explore</button>
                <button className={themeView === "list" ? "on" : ""} onClick={() => setThemeView("list")}>List</button>
              </div>
            )}
            <button className="primary" onClick={buildMap} disabled={buildingMap || !collIds.length}>
              {buildingMap ? "Building…" : themeMap ? "Rebuild" : "Build map"}
            </button>
          </div>
        </div>

        {mapError && <div className="note-err" style={{ marginTop: 8 }}>{mapError}</div>}

        {!collIds.length && <div className="empty">Select a library in the Conversation tab first.</div>}

        {collIds.length > 0 && buildingMap && (
          <div className="empty">Reading your library and organizing it into themes… this uses your current model and can take a moment.</div>
        )}

        {collIds.length > 0 && !themeMap && !buildingMap && (
          <div className="empty">No map yet — click <b>Build map</b> to organize {collLabel} into browsable themes.</div>
        )}

        {themeMap && !buildingMap && themeView === "explore" && renderExplore()}

        {themeMap && !buildingMap && themeView === "list" && (
          <div className="theme-list">
            {themeMap.themes.map((th, i) => (
              <div className="theme" key={i}>
                <button className="theme-h" onClick={() => setOpenThemes((o) => ({ ...o, [i]: !o[i] }))}>
                  <span className="caret">{openThemes[i] ? "▾" : "▸"}</span> {th.name}
                  <span className="muted"> · {th.subthemes.length}</span>
                </button>
                {openThemes[i] && (
                  <div className="sub-list">
                    {th.subthemes.length === 0 && (
                      <div className="angles">
                        {ANGLES.map((a) => (
                          <button key={a.label} className="angle" disabled={busy} onClick={() => askTheme(th.name, "", a)}>
                            {a.label}
                          </button>
                        ))}
                      </div>
                    )}
                    {th.subthemes.map((s, j) => (
                      <div className="sub" key={j}>
                        <div className="sub-name">{s.name}</div>
                        {s.blurb && <div className="sub-blurb muted">{s.blurb}</div>}
                        <div className="angles">
                          {ANGLES.map((a) => (
                            <button key={a.label} className="angle" disabled={busy} onClick={() => askTheme(th.name, s.name, a)} title={a.q(`"${s.name}"`)}>
                              {a.label}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="app">
      {/* Activity rail — the side navigator */}
      <div className="rail">
        <button
          className="rail-btn toggle"
          onClick={() => setSidebarOpen((v) => !v)}
          title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="4" width="18" height="16" rx="2" />
            <line x1="9" y1="4" x2="9" y2="20" />
          </svg>
        </button>
        <div className="rail-tabs">
          <button
            className={"rail-btn" + (mainTab === "chat" ? " active" : "")}
            onClick={() => setMainTab("chat")}
            title="Conversations"
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
            </svg>
            <span className="rail-label">Chat</span>
          </button>
          <button
            className={"rail-btn" + (mainTab === "themes" ? " active" : "")}
            onClick={() => setMainTab("themes")}
            title="Themes — a map of your library"
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3l1.9 4.6L18.5 9.5 14.8 12.8 15.7 18 12 15.3 8.3 18l.9-5.2L5.5 9.5l4.6-1.9z" />
            </svg>
            <span className="rail-label">Themes</span>
          </button>
        </div>
        <span className="rail-spacer" />
        <button className="rail-btn" onClick={() => setToolsOpen(true)} title="Settings — collections, providers, retrieval">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
          <span className="rail-label">Settings</span>
        </button>
      </div>

      {/* Sidebar (collapsible): conversations, or theme outline */}
      {sidebarOpen && (
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
      )}

      {/* Chat column */}
      <div className="main">
        {toolsOpen && settings && (
          <div className="tools-overlay" onClick={() => setToolsOpen(false)}>
            <div className="tools-modal" onClick={(e) => e.stopPropagation()}>
              <div className="tools-head">
                <b>Settings</b>
                <button className="ghost" onClick={() => setToolsOpen(false)} title="Close">
                  ✕
                </button>
              </div>
              <div className="tools-cols">
                <nav className="tools-nav">
                  {TOOLS_TABS.map(([id, label]) => (
                    <button key={id} className={toolsTab === id ? "active" : ""} onClick={() => setToolsTab(id)}>
                      {label}
                    </button>
                  ))}
                </nav>
                <div className="tools-body">
                  {toolsTab === "collections" && renderCollectionsTab()}
                  {toolsTab === "synthesis" && renderSynthesisTab()}
                  {toolsTab === "retrieval" && renderRetrievalTab()}
                  {toolsTab === "indexing" && renderIndexingTab()}
                  {toolsTab === "general" && renderGeneralTab()}
                  {toolsTab === "help" && renderHelpTab()}
                </div>
              </div>
              {toolsTab !== "collections" && toolsTab !== "help" && (
                <div className="tools-foot">
                  <button className="primary" onClick={saveSettings}>
                    Save
                  </button>
                  {settingsNote && (
                    <span className={settingsNote.startsWith("Error") ? "note-err" : "note-ok"}>{settingsNote}</span>
                  )}
                  <span className="muted" style={{ marginLeft: "auto", fontSize: 11 }}>
                    Changes apply to the next question.
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {mainTab === "themes" && renderThemesView()}

        {mainTab === "chat" && (
          <>
        {/* Transcript */}
        <div ref={scrollRef} className="transcript">
          {messages.length === 0 && (
            <div className="empty">Ask a question to start a conversation grounded in your library.</div>
          )}
          {messages.map((msg, idx) =>
            msg.role === "user" ? (
              <div key={idx} className="turn user">
                <div className="bubble-user">{msg.content}</div>
                <div className="msg-tools user-tools">
                  <button className="mini" onClick={() => copyText(msg.content, idx)}>
                    {copiedIdx === idx ? "Copied ✓" : "Copy"}
                  </button>
                </div>
              </div>
            ) : (
              <div key={idx} className="turn">
                {msg.thinking &&
                  (() => {
                    // Auto-expand the chain-of-thought while it streams (no answer
                    // yet on the last turn); the user's manual toggle overrides it.
                    const reasoningLive = busy && idx === messages.length - 1 && !msg.content;
                    const open = thinkOpen[idx] ?? reasoningLive;
                    return (
                      <div className="thinking">
                        <button
                          className="thinking-toggle"
                          onClick={() => setThinkOpen((o) => ({ ...o, [idx]: !(o[idx] ?? reasoningLive) }))}
                        >
                          <span className="caret">{open ? "▾" : "▸"}</span> Chain of thought
                          {reasoningLive && <span className="think-live"> · reasoning live…</span>}
                        </button>
                        {open && (
                          <div className="thinking-body" ref={idx === messages.length - 1 ? thinkRef : undefined}>
                            {msg.thinking}
                          </div>
                        )}
                      </div>
                    );
                  })()}
                <div className="card-assistant">
                  {msg.content ? (
                    renderRich(msg.content, msg.sources)
                  ) : (
                    <span className="muted think-live">{msg.thinking ? "Reasoning…" : "Thinking…"}</span>
                  )}
                </div>
                {msg.content && (
                  <div className="actions">
                    <button className="mini" onClick={() => copyText(msg.content, idx)}>
                      {copiedIdx === idx ? "Copied ✓" : "Copy"}
                    </button>
                    {idx === messages.length - 1 && (
                      <button className="mini" onClick={() => retryFrom(idx)} disabled={busy} title="Regenerate this answer">
                        ↻ Retry
                      </button>
                    )}
                    {msg.sources.length > 0 && (
                      <button className="mini" onClick={() => saveArtifact(idx)} title="Save this answer + sources as Markdown">
                        Save as Markdown
                      </button>
                    )}
                    {savedByIdx[idx] && (
                      <span className={savedByIdx[idx].startsWith("Error") ? "note-err" : "note-ok"}>
                        {savedByIdx[idx].startsWith("Error") ? savedByIdx[idx] : `Saved → ${savedByIdx[idx]}`}
                      </span>
                    )}
                  </div>
                )}
                {msg.sources.length > 0 && (
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
          <div className="composer-bar">
            {/* Left: which library/collections to search */}
            <div className="bar-left">
              <div className="coll-picker">
                <button onClick={() => setShowCollPicker((v) => !v)} title="Choose one or more collections to search">
                  📚 {collLabel} ▴
                </button>
                {showCollPicker && (
                  <div className="coll-menu up" onMouseLeave={() => setShowCollPicker(false)}>
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
            </div>

            {/* Right: provider, model, status, tokens, panels */}
            <div className="bar-right">
              <span className="tokens" title="Tokens this conversation · input / output">
                ↑ {tokens.in.toLocaleString()} · ↓ {tokens.out.toLocaleString()}
              </span>
              <select
                value={settings?.llm_provider ?? "ollama"}
                onChange={(e) => chooseProvider(e.target.value)}
                title="Synthesis provider (configure keys in Settings)"
              >
                {readyProviders().map((id) => (
                  <option key={id} value={id}>
                    {providerLabel(id)}
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
              {llmStatus && (
                <span className="llm-status" title={llmStatus.message} onClick={() => checkLlm(model)}>
                  <span className={"dot " + (llmStatus.ok ? "ok" : "bad")} />
                </span>
              )}
            </div>
          </div>
        </div>
          </>
        )}
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
          {reader.missing ? (
            <div className="reader-missing">
              <h3>Source not found</h3>
              <p>
                This book isn't where it was when it was indexed — it was likely moved or renamed:
              </p>
              <code className="missing-path">{reader.path}</code>
              <p>
                Point the collection at the book's new location: open <b>Settings → Collections</b>,
                add the new folder (and remove the old one), then re-index so citations resolve again.
              </p>
              <button
                className="primary"
                onClick={() => {
                  setReader(null);
                  setToolsTab("collections");
                  setToolsOpen(true);
                }}
              >
                Open Settings → Collections
              </button>
            </div>
          ) : (
            <iframe key={readerSrc} title="source" src={readerSrc} />
          )}
        </div>
      )}
    </div>
  );
}
