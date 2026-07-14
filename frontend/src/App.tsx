import { Fragment, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import PdfReader from "./PdfReader";
import { extOf, EXT_FAMILY } from "./generated/supportedExts";

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
// Mirrors ls_llm::PromptMeta — what actually went into the prompt for one ask.
type PromptMeta = {
  notes_injected: boolean;
  notes_tokens: number;
  notes_truncated: boolean;
  digest_lines: number;
  recent_turns: number;
  dropped_turns: number;
  prompt_tokens: number;
};
type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  thinking: string;
  sources: Src[];
  loose?: boolean;
  ctx?: PromptMeta;
};
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

type ToolsTab = "collections" | "synthesis" | "retrieval" | "memory" | "indexing" | "general" | "help";
const TOOLS_TABS: [ToolsTab, string][] = [
  ["collections", "Collections"],
  ["synthesis", "Synthesis"],
  ["retrieval", "Retrieval"],
  ["memory", "Memory"],
  ["indexing", "Indexing"],
  ["general", "General"],
  ["help", "Help"],
];

// Mirrors ls_llm::estimate_tokens (chars/4 Latin, /2.5 substantially-Cyrillic) —
// cosmetic counter only; the authoritative numbers come back in PromptMeta.
const NOTES_TOKEN_CAP = 600;
function estimateTokens(s: string): number {
  const total = [...s].length;
  if (total === 0) return 0;
  let cyr = 0;
  for (const c of s) if (c >= "Ѐ" && c <= "ӿ") cyr++;
  return Math.ceil(total / (cyr * 5 >= total ? 2.5 : 4));
}

type SubTheme = { name: string; blurb: string };
type Theme = { name: string; subthemes: SubTheme[] };
type ThemeMap = { generated_at: number; model: string; book_count: number; themes: Theme[] };
type CatalogBook = { title: string; author: string; source_path: string; format: string; chunks: number };
type CatalogEntry = { label: string; book: string; page: number | null; source_path: string };
type LibraryCatalog = { books: CatalogBook[]; index: CatalogEntry[] };

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

type ReaderKind = "pdf" | "md" | "other";
type Reader = {
  path: string;
  page: number | null;
  missing: boolean;
  kind: ReaderKind;
  /// The cited passage (for scroll/highlight in md, and shown for non-renderable formats).
  citeText?: string;
  /// Loaded Markdown content (kind === "md").
  text?: string;
  error?: string;
  /// PDF.js couldn't open this file — fall back to the native iframe viewer.
  pdfNative?: boolean;
  /// Text source exceeded the read cap; only the first window is shown.
  truncated?: boolean;
  totalBytes?: number;
};

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
  gpu_device: string;
  hybrid_top_k: number;
  final_top_k: number;
  min_relevance: number;
  memory_enabled: boolean;
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
  by_format?: Record<string, [number, number]>;
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
  const [readerFull, setReaderFull] = useState(false);

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [convId, setConvId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [dataSafety, setDataSafety] = useState<{ at_risk: boolean; provider: string; path: string } | null>(null);
  const [safetyDismissed, setSafetyDismissed] = useState(false);
  const [savedByIdx, setSavedByIdx] = useState<Record<number, string>>({});
  const [thinkOpen, setThinkOpen] = useState<Record<number, boolean>>({});
  const [tokens, setTokens] = useState({ in: 0, out: 0 });
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const [notedIdx, setNotedIdx] = useState<number | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const [toolsOpen, setToolsOpen] = useState(false);
  const [toolsTab, setToolsTab] = useState<ToolsTab>("collections");
  const [settings, setSettings] = useState<Settings | null>(null);
  // The user's global notebook (Settings → Memory); loaded when the tab opens.
  const [note, setNote] = useState("");
  const [noteStatus, setNoteStatus] = useState<string | null>(null);
  const [noteUpdatedAt, setNoteUpdatedAt] = useState(0); // unix secs; 0 = never
  // Re-index nudge: books chunked by an older scheme in the current collection.
  const [legacyBooks, setLegacyBooks] = useState(0);
  const [nudgeDismissed, setNudgeDismissed] = useState(false);
  const [rechunkNote, setRechunkNote] = useState<string | null>(null);
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
  const [themeView, setThemeView] = useState<"explore" | "list" | "titles" | "index">("explore");
  // Library catalog (Titles browser + library-wide Index), cached per selection.
  const [catalog, setCatalog] = useState<LibraryCatalog | null>(null);
  const [catalogFor, setCatalogFor] = useState("");
  const [catFilter, setCatFilter] = useState("");
  const [indexLetter, setIndexLetter] = useState("A");
  // The input stays instant; filtering runs against a deferred value so a
  // keystroke never blocks on re-filtering/rendering 62k index entries.
  const catQuery = useDeferredValue(catFilter);
  const [mapProgress, setMapProgress] = useState<string | null>(null);
  const [exploreTree, setExploreTree] = useState<BNode[]>([]);
  const [focusPath, setFocusPath] = useState<number[]>([]);
  const [deepening, setDeepening] = useState(false);
  const [llmStatus, setLlmStatus] = useState<{ ok: boolean; message: string } | null>(null);

  // Manage operates on the first selected collection.
  const currentColl = collections.find((c) => c.id === collIds[0]) || null;
  // A fresh user has no collection with source folders yet — show onboarding
  // instead of the dead "ask a question" empty-state (the composer is disabled
  // with nothing to search, which is a dead end otherwise).
  const hasUsableLibrary = collections.some((c) => c.source_paths.length > 0);
  // Conversation history is scoped to the selected library: show only chats
  // that involve at least one selected collection (legacy chats with no
  // collections recorded stay visible everywhere).
  const visibleConvs = conversations.filter(
    (c) => !c.collection_ids.length || c.collection_ids.some((id) => collIds.includes(id))
  );
  const collLabel =
    collIds.length === 0
      ? "Select collections"
      : collIds.length === 1
        ? collections.find((c) => c.id === collIds[0])?.name ?? "1 collection"
        : `${collIds.length} collections`;
  const scrollRef = useRef<HTMLDivElement>(null);
  const mdReaderRef = useRef<HTMLDivElement>(null);
  // Progressive text rendering (M1b): how many slices of the current md
  // source are mounted, and a tick that re-arms the cite-scroll effect once
  // the slice containing the citation has rendered.
  const [mdSlices, setMdSlices] = useState(1);
  const [mdTick, setMdTick] = useState(0);
  const logRef = useRef<HTMLPreElement>(null);
  const thinkRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    invoke<Collection[]>("list_collections").then(setCollections).catch(console.error);
    invoke<Conversation[]>("list_conversations").then(setConversations).catch(console.error);
    invoke<{ at_risk: boolean; provider: string; path: string }>("data_safety").then(setDataSafety).catch(console.error);
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

  // When a Markdown source loads, scroll to (and briefly highlight) the cited
  // passage: find the first rendered text node containing the passage's prefix.
  useEffect(() => {
    const host = mdReaderRef.current;
    const cite = reader?.citeText;
    if (!host || !reader?.text || !cite) return;
    const norm = (x: string) => x.replace(/\s+/g, " ").trim().toLowerCase();
    const needle = norm(cite).slice(0, 60);
    if (!needle) return;
    const walker = document.createTreeWalker(host, NodeFilter.SHOW_TEXT);
    let node: Node | null;
    while ((node = walker.nextNode())) {
      if (norm(node.textContent ?? "").includes(needle.slice(0, 40))) {
        const el = node.parentElement;
        if (el) {
          el.scrollIntoView({ block: "center" });
          el.classList.add("cite-flash");
          setTimeout(() => el.classList.remove("cite-flash"), 2500);
        }
        break;
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reader?.text, mdTick]);

  // Slice boundaries for progressive rendering: files over 2 MiB render in
  // ~512 KiB paragraph-aligned steps per idle callback so the WKWebView main
  // thread never builds a many-MiB DOM synchronously.
  const MD_PROGRESSIVE_MIN = 2 * 1024 * 1024;
  const MD_SLICE = 512 * 1024;
  const mdSliceOffsets = useMemo(() => {
    const text = reader?.text ?? "";
    if (text.length <= MD_PROGRESSIVE_MIN) return null;
    const offs: number[] = [0];
    let at = MD_SLICE;
    while (at < text.length) {
      const nl = text.indexOf("\n\n", at);
      const cut = nl === -1 ? text.length : nl;
      offs.push(cut);
      at = cut + MD_SLICE;
    }
    offs.push(text.length);
    return offs;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reader?.text]);
  useEffect(() => {
    // Reset slice progress when a new text source loads; fast-forward past
    // the slice containing the cited passage so the cite-jump can land.
    if (!mdSliceOffsets) {
      setMdSlices(1);
      return;
    }
    let initial = 1;
    const cite = reader?.citeText;
    if (cite && reader?.text) {
      const probe = cite.slice(0, 40).trim();
      const at = probe ? reader.text.indexOf(probe) : -1;
      if (at >= 0) {
        while (initial < mdSliceOffsets.length - 1 && mdSliceOffsets[initial] < at) initial++;
      }
    }
    setMdSlices(initial);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mdSliceOffsets]);
  useEffect(() => {
    if (!mdSliceOffsets || mdSlices >= mdSliceOffsets.length - 1) {
      setMdTick((t) => t + 1);
      return;
    }
    // requestIdleCallback where available; WebKitGTK builds may lack it.
    const ric: (cb: () => void) => number =
      "requestIdleCallback" in window
        ? (cb) => (window as unknown as { requestIdleCallback: (cb: () => void) => number }).requestIdleCallback(cb)
        : (cb) => window.setTimeout(cb, 0);
    const id = ric(() => setMdSlices((n) => n + 1));
    return () => {
      if ("cancelIdleCallback" in window) {
        (window as unknown as { cancelIdleCallback: (id: number) => void }).cancelIdleCallback(id);
      } else {
        clearTimeout(id);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mdSliceOffsets, mdSlices]);

  // Reader view (full screen): leave it when the reader closes; Esc exits.
  // WKWebView only delivers Escape when a focusable element has focus, so on
  // entry we focus the reader element itself (tabIndex=-1) and listen there;
  // the window listener is a fallback.
  const readerElRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!reader) setReaderFull(false);
  }, [reader]);
  useEffect(() => {
    if (!readerFull) return;
    readerElRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setReaderFull(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [readerFull]);

  // Load the notebook when the Memory tab opens.
  useEffect(() => {
    if (!toolsOpen || toolsTab !== "memory") return;
    invoke<{ content: string; updated_at: number }>("get_note_info", { scope: "global" })
      .then((i) => {
        setNote(i.content);
        setNoteUpdatedAt(i.updated_at);
      })
      .catch(console.error);
    setNoteStatus(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toolsOpen, toolsTab]);

  // Switching library hides conversations outside it — if the active one is
  // hidden, drop to a fresh chat so the transcript matches the list.
  useEffect(() => {
    if (!convId) return;
    const cur = conversations.find((c) => c.id === convId);
    if (cur && cur.collection_ids.length && !cur.collection_ids.some((id) => collIds.includes(id))) {
      newChat();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collIds]);

  // Load the library catalog when the Titles/Index views open (cached per selection).
  useEffect(() => {
    const key = collIds.join(",");
    if (mainTab !== "themes" || (themeView !== "titles" && themeView !== "index")) return;
    if (!collIds.length || (catalog && catalogFor === key)) return;
    invoke<LibraryCatalog>("library_catalog", { collectionIds: collIds })
      .then((c) => {
        setCatalog(c);
        setCatalogFor(key);
      })
      .catch(console.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mainTab, themeView, collIds]);

  // Index health for the passive re-index nudge (manifest-only, no folder scan).
  useEffect(() => {
    if (!toolsOpen || toolsTab !== "collections" || !currentColl) return;
    invoke<{ legacy_books: number }>("index_health", { collectionId: currentColl.id })
      .then((h) => setLegacyBooks(h.legacy_books))
      .catch(console.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toolsOpen, toolsTab, currentColl?.id, indexing]);

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
    // Provenance: mark the in-flight answer as lower-confidence when it came from
    // the fuzzy fallback tier.
    const unProv = listen<boolean>("ask-provenance", (e) =>
      setMessages((prev) => {
        if (!prev.length) return prev;
        const last = prev.length - 1;
        if (prev[last].role !== "assistant") return prev;
        const copy = [...prev];
        copy[last] = { ...copy[last], loose: e.payload };
        return copy;
      })
    );
    // Context used for this answer (notes/digest/turns) — from the prompt builder.
    const unCtx = listen<PromptMeta>("ask-context", (e) =>
      setMessages((prev) => {
        if (!prev.length) return prev;
        const last = prev.length - 1;
        if (prev[last].role !== "assistant") return prev;
        const copy = [...prev];
        copy[last] = { ...copy[last], ctx: e.payload };
        return copy;
      })
    );
    // Live progress for the (potentially multi-minute) theme-map build.
    const unMap = listen<string>("map-progress", (e) => setMapProgress(e.payload));
    return () => {
      unTok.then((f) => f());
      unThink.then((f) => f());
      unUsage.then((f) => f());
      unProv.then((f) => f());
      unCtx.then((f) => f());
      unMap.then((f) => f());
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
        // Per-format legibility for the first post-upgrade re-scope run:
        // "indexed 12 md, 3 txt · skipped 480 pdf (up-to-date)".
        const fmt = Object.entries(s.by_format ?? {});
        const indexedBits = fmt.filter(([, v]) => v[0] > 0).map(([k, v]) => `${v[0]} ${k}`);
        const skippedBits = fmt.filter(([, v]) => v[1] > 0).map(([k, v]) => `${v[1]} ${k}`);
        const detail =
          indexedBits.length || skippedBits.length
            ? ` (${[
                indexedBits.length ? `indexed ${indexedBits.join(", ")}` : "",
                skippedBits.length ? `skipped ${skippedBits.join(", ")}` : "",
              ]
                .filter(Boolean)
                .join(" · ")})`
            : "";
        setIndexNote(
          `Done — ${s.books_indexed} indexed, ${s.books_unchanged} unchanged, ${s.books_skipped + s.books_failed} skipped, ${s.chunks_written} chunks written.${detail}`
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

  function openTools(tab: ToolsTab) {
    setToolsTab(tab);
    setToolsOpen(true);
  }

  // First-run onboarding: a fresh user has no indexed library, so the composer is
  // disabled and the passive empty-state is a dead end. Route them straight to the
  // two things needed for a first answer — add+index a folder, and pick a model.
  function renderOnboarding() {
    const hasFolders = collections.some((c) => c.source_paths.length > 0);
    const modelReady = !!llmStatus?.ok;
    return (
      <div className="onboarding">
        <div className="onboarding-card">
          <div className="onboarding-title">Welcome to LibSearch Studio</div>
          <div className="onboarding-sub">
            Chat with your own PDF &amp; ebook library — answers are grounded in your books with
            clickable citations, and nothing leaves your machine at query time. Two quick steps to
            your first answer:
          </div>
          <ol className="onboarding-steps">
            <li className={hasFolders ? "done" : ""}>
              <div className="step-head">
                <span className="step-num">{hasFolders ? "✓" : "1"}</span>
                <b>Add a folder of books and index it</b>
              </div>
              <div className="step-body">
                Point LibSearch at a folder of PDFs/EPUBs; indexing builds the searchable index
                locally (a GPU option is available for large libraries).
                <div>
                  <button className="primary" onClick={() => openTools("collections")}>
                    {hasFolders ? "Manage library →" : "Add a library →"}
                  </button>
                </div>
              </div>
            </li>
            <li className={modelReady ? "done" : ""}>
              <div className="step-head">
                <span className="step-num">{modelReady ? "✓" : "2"}</span>
                <b>Choose where answers come from</b>
              </div>
              <div className="step-body">
                Use local Ollama (fully offline) or add a cloud provider key. This is the model that
                writes the grounded answer.
                <div>
                  <button onClick={() => openTools("synthesis")}>Pick a model →</button>
                </div>
              </div>
            </li>
          </ol>
          <div className="onboarding-foot muted">
            New here? The <button className="linklike" onClick={() => openTools("help")}>Help</button>{" "}
            tab explains how retrieval and citations work.
          </div>
        </div>
      </div>
    );
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

  // Append an answer to the global notebook (explicit user action — the only
  // way the app ever writes memory).
  async function addToNotes(text: string, idx: number) {
    try {
      const cur = await invoke<string>("get_note", { scope: "global" });
      const next = cur.trim() ? `${cur.trimEnd()}\n\n---\n${text.trim()}` : text.trim();
      await invoke("set_note", { scope: "global", content: next });
      setNote(next); // keep the Memory tab in sync if it's open later
      setNotedIdx(idx);
      setTimeout(() => setNotedIdx((c) => (c === idx ? null : c)), 1500);
    } catch (e) {
      console.error(e);
    }
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
    setMapProgress(null);
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
    // Reader-kind routing via the generated extension map (the canonical rule
    // lives in ls-core; last-dot splitting would route ".fb2.zip" wrong).
    // PDFs render in PdfReader; Markdown/plain text in-app; anything else
    // (epub/mobi/…) gets a passage view + external open.
    const family = EXT_FAMILY[extOf(s.source_path) ?? ""] ?? "other";
    const kind: ReaderKind = family === "pdf" ? "pdf" : family === "md" || family === "txt" ? "md" : "other";
    setReader({ path: s.source_path, page: s.page, missing: false, kind, citeText: s.text });
    // The book may have been moved/renamed since indexing — warn instead of
    // showing a silently-blank reader.
    try {
      const ok = await invoke<boolean>("source_exists", { path: s.source_path });
      setReader((r) => (r && r.path === s.source_path ? { ...r, missing: !ok } : r));
      if (!ok) return;
    } catch {
      /* leave as-is */
    }
    if (kind === "md") {
      try {
        const st = await invoke<{ text: string; truncated: boolean; total_bytes: number }>(
          "read_source_text",
          { path: s.source_path }
        );
        setReader((r) =>
          r && r.path === s.source_path
            ? { ...r, text: st.text, truncated: st.truncated, totalBytes: st.total_bytes }
            : r
        );
      } catch (e) {
        setReader((r) => (r && r.path === s.source_path ? { ...r, error: String(e) } : r));
      }
    }
  }

  async function pickFolder(): Promise<string | null> {
    const dir = await open({ directory: true, multiple: false, title: "Choose a folder of books & documents" });
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
    type Block =
      | { type: "p"; text: string }
      | { type: "ul" | "ol"; items: string[] }
      | { type: "h"; level: number; text: string }
      | { type: "code"; text: string };
    const blocks: Block[] = [];
    let para: string[] = [];
    let list: { type: "ul" | "ol"; items: string[] } | null = null;
    let code: string[] | null = null;
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
      // Fenced code: verbatim until the closing fence (monospace, no highlight).
      if (code !== null) {
        if (line.trim().startsWith("```")) {
          blocks.push({ type: "code", text: code.join("\n") });
          code = null;
        } else {
          code.push(raw);
        }
        continue;
      }
      if (line.trim().startsWith("```")) {
        flushPara();
        flushList();
        code = [];
        continue;
      }
      const heading = line.match(/^\s*(#{1,4})\s+(.*)$/);
      const bullet = line.match(/^\s*[-*]\s+(.*)$/);
      const numbered = line.match(/^\s*\d+\.\s+(.*)$/);
      if (heading) {
        flushPara();
        flushList();
        blocks.push({ type: "h", level: heading[1].length, text: heading[2] });
      } else if (bullet) {
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
    if (code !== null) blocks.push({ type: "code", text: code.join("\n") });
    flushPara();
    flushList();

    return blocks.map((b, i) => {
      if (b.type === "h") {
        const H = (["h1", "h2", "h3", "h4"] as const)[b.level - 1];
        return <H key={i}>{renderInline(b.text, sources)}</H>;
      }
      if (b.type === "code")
        return (
          <pre key={i} className="rich-code">
            <code>{b.text}</code>
          </pre>
        );
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
            {legacyBooks > 0 && !nudgeDismissed && !rechunkNote && (
              <div className="nudge">
                <span>
                  {legacyBooks} book(s) were indexed with an older chunking scheme. Answers work fine, but re-chunking
                  improves passage boundaries and citation pages.
                </span>
                <button className="mini" onClick={() => rechunkCollection(currentColl)} disabled={indexing}>
                  Re-chunk on next Index
                </button>
                <button className="ghost" title="Dismiss for now" onClick={() => setNudgeDismissed(true)}>
                  ✕
                </button>
              </div>
            )}
            {rechunkNote && <div className="note-ok" style={{ marginTop: 6, fontSize: 12.5 }}>{rechunkNote}</div>}
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

  // Explicit re-chunk opt-in: forget the collection's fingerprints so the next
  // Index run re-embeds everything with the current chunker. Never automatic —
  // a full re-embed of a large library takes hours on the GPU.
  async function rechunkCollection(coll: Collection) {
    const ok = window.confirm(
      `Re-chunk "${coll.name}"?\n\nThe next Index / Re-index will re-embed ALL its books with the current chunker ` +
        `(better passage boundaries + citation pages). A large library can take hours on the GPU. Nothing is deleted now.`
    );
    if (!ok) return;
    try {
      const n = await invoke<number>("reset_chunker_state", { collectionId: coll.id });
      setLegacyBooks(0);
      setRechunkNote(`Ready — click Index / Re-index to re-embed ${n} book(s) with the current chunker.`);
    } catch (e) {
      setRechunkNote("Error: " + String(e));
    }
  }

  async function saveNote() {
    try {
      await invoke("set_note", { scope: "global", content: note });
      setNoteUpdatedAt(Math.floor(Date.now() / 1000));
      setNoteStatus("Saved ✓");
    } catch (e) {
      setNoteStatus("Error: " + String(e));
    }
  }

  async function exportNote() {
    try {
      const path = await invoke<string>("export_note", { scope: "global" });
      setNoteStatus(`Exported → ${path}`);
    } catch (e) {
      setNoteStatus("Error: " + String(e));
    }
  }

  function renderMemoryTab() {
    if (!settings) return null;
    const tokens = estimateTokens(note);
    const over = tokens > NOTES_TOKEN_CAP;
    return (
      <div>
        <div className="tools-note muted" style={{ marginBottom: 10 }}>
          Your notebook is the app's entire memory — nothing is remembered unless you write it here. It's given to the
          model as background context on every question (never as a citable source). Standing preferences work well:
          "Prefer concise answers", "I'm studying for a systems-design interview", "Answer in Russian when I ask in
          Russian".
        </div>
        <label className="row" style={{ gap: 8, marginBottom: 8 }}>
          <input
            type="checkbox"
            checked={settings.memory_enabled}
            onChange={(e) => editSetting("memory_enabled", e.target.checked)}
          />
          <span>Use my notes when answering (off = notes are kept but never sent to the model)</span>
        </label>
        {(() => {
          // Staleness cue: standing notes silently steer every answer, so nudge a
          // review when they haven't been touched in ~3 months.
          const days = noteUpdatedAt > 0 ? Math.floor((Date.now() / 1000 - noteUpdatedAt) / 86400) : 0;
          return note.trim() && days > 90 ? (
            <div className="nudge" style={{ marginBottom: 8 }}>
              <span>
                These notes were last updated {days} days ago and still shape every answer — worth a quick
                re-read?
              </span>
            </div>
          ) : null;
        })()}
        <textarea
          className="note-editor"
          value={note}
          onChange={(e) => {
            setNote(e.target.value);
            setNoteStatus(null);
          }}
          placeholder="Anything you want the model to keep in mind, in your own words…"
          rows={12}
        />
        <div className="row" style={{ gap: 8, alignItems: "center", marginTop: 8 }}>
          <button className="primary" onClick={saveNote}>
            Save notes
          </button>
          <button onClick={exportNote} title="Write the notebook to a Markdown file in your artifacts folder">
            Export .md
          </button>
          <span className={"muted" + (over ? " note-over" : "")} style={{ fontSize: 12 }}>
            ~{tokens} / {NOTES_TOKEN_CAP} tokens injected{over ? " — over the cap; the end will be trimmed" : ""}
          </span>
          {noteStatus && (
            <span className={noteStatus.startsWith("Error") ? "note-err" : "note-ok"} style={{ fontSize: 12 }}>
              {noteStatus}
            </span>
          )}
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
          <label>Fast index · device</label>
          <input
            placeholder="mps (Apple GPU) · cuda · cpu"
            value={settings.gpu_device ?? "mps"}
            onChange={(e) => editSetting("gpu_device", e.target.value)}
            spellCheck={false}
          />
        </div>
        <div style={{ marginTop: 10 }}>
          <button onClick={runSetup} disabled={settingUp} title="Create a local venv, install deps, and download/export the search models">
            {settingUp ? "Setting up…" : "Set up search models (auto)"}
          </button>
          <div className="muted" style={{ marginTop: 6 }}>
            Required to index &amp; search: downloads the ONNX embedding models (and enables GPU
            indexing) into a local venv (several GB). Ready to index &amp; ask as soon as it finishes —
            no restart needed.
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
        <label>Data folder</label>
        <div className="row">
          <button onClick={revealDataFolder}>Reveal in file manager</button>
          <span className="muted" style={{ fontSize: 12 }}>
            Holds your index, history &amp; settings. To back up: quit the app, copy this folder.
            Keep it off Dropbox/iCloud — sync corrupts the index.
          </span>
        </div>
      </div>
    );
  }

  async function revealDataFolder() {
    await invoke("reveal_data_folder").catch((e) => setSettingsNote("Error: " + String(e)));
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

  // Lowercased search fields computed ONCE per catalog load — not per keystroke
  // (62k entries × 2 toLowerCase per keystroke froze the Index search).
  const searchableIndex = useMemo(
    () =>
      (catalog?.index ?? []).map((e) => ({
        e,
        low: e.label.toLowerCase() + "\u0000" + e.book.toLowerCase(),
      })),
    [catalog]
  );
  const searchableBooks = useMemo(
    () =>
      (catalog?.books ?? []).map((b) => ({
        b,
        low: b.title.toLowerCase() + "\u0000" + b.author.toLowerCase(),
      })),
    [catalog]
  );

  // Never render more than this many rows — a broad query ("a") matches tens of
  // thousands of entries and mounting them all freezes the webview.
  const RESULT_CAP = 800;

  // First grouping letter for A–Z browsing (non-letters bucket under '#').
  function groupLetter(x: string): string {
    const c = (x.trim()[0] ?? "#").toUpperCase();
    return /[A-ZА-ЯЁ]/.test(c) ? c : "#";
  }

  function renderTitles() {
    if (!catalog) return <div className="empty">Loading titles…</div>;
    const q = catQuery.trim().toLowerCase();
    const books: CatalogBook[] = [];
    let matched = 0;
    for (const { b, low } of searchableBooks) {
      if (!q || low.includes(q)) {
        matched++;
        if (books.length < RESULT_CAP) books.push(b);
      }
    }
    let lastLetter = "";
    return (
      <div className="catalog">
        <div className="catalog-bar">
          <input
            placeholder={`Filter ${catalog.books.length} titles…`}
            value={catFilter}
            onChange={(e) => setCatFilter(e.target.value)}
            spellCheck={false}
            autoCorrect="off"
            autoCapitalize="off"
          />
          <span className="muted">
            {matched > books.length ? `first ${books.length} of ${matched} — type to narrow` : `${matched} shown`}
          </span>
        </div>
        <div className="catalog-list">
          {books.map((b) => {
            const letter = groupLetter(b.title);
            const header = letter !== lastLetter;
            lastLetter = letter;
            return (
              <Fragment key={b.title}>
                {header && <div className="cat-letter">{letter}</div>}
                <div className="cat-row">
                  <span className="cat-title" title={b.source_path}>
                    {b.title}
                    {b.author && <span className="muted"> — {b.author}</span>}
                  </span>
                  <span className="muted cat-meta">
                    {b.format} · {b.chunks} passages
                  </span>
                  <button
                    className="mini"
                    onClick={() => askNew(`Give me an overview of "${b.title}" — what does it cover and what are its key ideas?`)}
                    disabled={busy}
                    title="Ask the library about this book"
                  >
                    Ask
                  </button>
                  <button
                    className="mini"
                    onClick={() => openSource({ rank: 0, citation: b.title, source_path: b.source_path, page: 1 })}
                    title="Open the book"
                  >
                    Open
                  </button>
                  <button
                    className="mini"
                    onClick={async () => {
                      // §4.2: the sanctioned way to chapter-enrich or restamp a
                      // legacy book — deletes and re-embeds this ONE book.
                      if (
                        !window.confirm(
                          `Re-index "${b.title}"?\n\nThis removes its current chunks; the next Index run re-embeds just this book with the latest extractor (chapters, format).`
                        )
                      )
                        return;
                      try {
                        await invoke("reindex_book", {
                          collectionIds: collIds,
                          sourcePath: b.source_path,
                        });
                        setIndexNote(`"${b.title}" cleared — run Index to re-embed it.`);
                        // Invalidate the cached catalog so Titles/Index reload.
                        setCatalogFor("");
                        setCatalog(null);
                      } catch (e) {
                        setIndexNote(`Re-index failed: ${e}`);
                      }
                    }}
                    title="Delete this book's chunks and re-embed it on the next Index run"
                  >
                    ↻ Re-index
                  </button>
                </div>
              </Fragment>
            );
          })}
          {books.length === 0 && <div className="muted" style={{ padding: 12 }}>No titles match.</div>}
        </div>
      </div>
    );
  }

  function renderIndex() {
    if (!catalog) return <div className="empty">Building the index…</div>;
    const letters = Array.from(new Set(catalog.index.map((e) => groupLetter(e.label)))).sort();
    const q = catQuery.trim().toLowerCase();
    // Like a book's index: flip to a letter — or search across all of it.
    // Early-exit single pass, hard-capped: broad queries match tens of
    // thousands of entries and rendering them all froze the view.
    const entries: CatalogEntry[] = [];
    let matched = 0;
    for (const { e, low } of searchableIndex) {
      const hit = q ? low.includes(q) : groupLetter(e.label) === indexLetter;
      if (hit) {
        matched++;
        if (entries.length < RESULT_CAP) entries.push(e);
      }
    }
    return (
      <div className="catalog">
        <div className="catalog-bar">
          <input
            placeholder={`Search ${catalog.index.length} index entries…`}
            value={catFilter}
            onChange={(e) => setCatFilter(e.target.value)}
            spellCheck={false}
            autoCorrect="off"
            autoCapitalize="off"
          />
          <span className="muted">
            {matched > entries.length ? `first ${entries.length} of ${matched} — type to narrow` : `${matched} shown`}
          </span>
        </div>
        {!q && (
          <div className="letter-rail">
            {letters.map((l) => (
              <button key={l} className={l === indexLetter ? "on" : ""} onClick={() => setIndexLetter(l)}>
                {l}
              </button>
            ))}
          </div>
        )}
        <div className="catalog-list">
          {entries.map((e, i) => (
            <div
              key={`${e.label}·${e.book}·${i}`}
              className="cat-row idx"
              onClick={() => openSource({ rank: 0, citation: `${e.book} · ${e.label}`, source_path: e.source_path, page: e.page })}
              title="Open the book at this chapter"
            >
              <span className="cat-title">{e.label}</span>
              <span className="muted cat-meta">
                {e.book}
                {e.page ? ` · p.${e.page}` : ""}
              </span>
            </div>
          ))}
          {entries.length === 0 && <div className="muted" style={{ padding: 12 }}>Nothing here.</div>}
        </div>
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
                {/* Provenance of the CACHED map — the model that BUILT it, which can
                    differ from the currently selected one until a Rebuild. */}
                {themeMap.book_count} books · built {new Date(themeMap.generated_at).toLocaleString()} with{" "}
                {themeMap.model}
                {model && themeMap.model !== model && (
                  <> — current model is {model}; Rebuild to refresh</>
                )}
              </div>
            ) : (
              <div className="muted">A map of {collLabel} into themes you can explore — click an angle to launch a focused, grounded question.</div>
            )}
          </div>
          <div className="row">
            {collIds.length > 0 && (
              <div className="seg">
                <button className={themeView === "explore" ? "on" : ""} onClick={() => setThemeView("explore")}>Explore</button>
                <button className={themeView === "list" ? "on" : ""} onClick={() => setThemeView("list")}>List</button>
                <button className={themeView === "titles" ? "on" : ""} onClick={() => setThemeView("titles")}>Titles</button>
                <button className={themeView === "index" ? "on" : ""} onClick={() => setThemeView("index")}>Index</button>
              </div>
            )}
            <button
              className="primary"
              onClick={buildMap}
              disabled={buildingMap || !collIds.length}
              title={model ? `Build the map with the current model (${model})` : "Build the map"}
            >
              {buildingMap ? "Building…" : themeMap ? "Rebuild" : "Build map"}
            </button>
          </div>
        </div>

        {mapError && <div className="note-err" style={{ marginTop: 8 }}>{mapError}</div>}

        {!collIds.length && <div className="empty">Select a library in the Conversation tab first.</div>}

        {collIds.length > 0 && buildingMap && (themeView === "explore" || themeView === "list") && (
          <div className="empty">
            <div>Reading your library and organizing it into themes… slow models can take several minutes.</div>
            {mapProgress && <div className="muted" style={{ marginTop: 6 }}>{mapProgress}</div>}
            <div style={{ marginTop: 10 }}>
              <button onClick={() => invoke("cancel_map").catch(console.error)}>■ Cancel build</button>
            </div>
            <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>
              Titles and Index (above) work instantly — no need to wait.
            </div>
          </div>
        )}

        {collIds.length > 0 && !themeMap && !buildingMap && (themeView === "explore" || themeView === "list") && (
          <div className="empty">No map yet — click <b>Build map</b> to organize {collLabel} into browsable themes.</div>
        )}

        {collIds.length > 0 && themeView === "titles" && renderTitles()}
        {collIds.length > 0 && themeView === "index" && renderIndex()}

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
        {/* Library selector — scopes chat, themes, and the conversation list. */}
        <div className="sidebar-lib">
          <div className="coll-picker">
            <button onClick={() => setShowCollPicker((v) => !v)} title="Choose one or more collections to work with">
              📚 {collLabel} ▾
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
        </div>
        <div className="sidebar-head">
          <b>Conversations</b>
          <button className="ghost" onClick={newChat} title="Start a new conversation">
            + New
          </button>
        </div>
        <div className="conv-list">
          {visibleConvs.length === 0 && (
            <div className="muted" style={{ padding: "4px 10px" }}>No conversations in this library yet.</div>
          )}
          {visibleConvs.map((c) =>
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
        {dataSafety?.at_risk && !safetyDismissed && (
          <div className="safety-banner">
            <span>
              ⚠️ Your library index is on <b>{dataSafety.provider}</b> ({dataSafety.path}). Cloud sync can
              corrupt the index — move the data folder off it (Settings → General → Reveal).
            </span>
            <button className="linklike" onClick={() => openTools("general")}>Open Settings</button>
            <button className="ghost" title="Dismiss" onClick={() => setSafetyDismissed(true)}>✕</button>
          </div>
        )}
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
                  {toolsTab === "memory" && renderMemoryTab()}
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
          {messages.length === 0 &&
            (hasUsableLibrary ? (
              <div className="empty">Ask a question to start a conversation grounded in your library.</div>
            ) : (
              renderOnboarding()
            ))}
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
                {msg.loose && msg.content && (
                  <div className="provenance" title="No passage cleared the confidence threshold, so this used the best loosely-related matches.">
                    ◐ Answered from loosely-related passages — treat as lower confidence.
                  </div>
                )}
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
                    <button
                      className="mini"
                      onClick={() => addToNotes(msg.content, idx)}
                      title="Append this answer to your notebook (Settings → Memory)"
                    >
                      {notedIdx === idx ? "Noted ✓" : "+ Notes"}
                    </button>
                    {msg.ctx && (
                      <span
                        className="ctx-chip"
                        title={`Context used for this answer:\n• Notes: ${
                          msg.ctx.notes_injected
                            ? `~${msg.ctx.notes_tokens} tokens${msg.ctx.notes_truncated ? " (truncated to fit)" : ""}`
                            : "not used"
                        }\n• Recent turns: ${msg.ctx.recent_turns}\n• Earlier-topics digest: ${
                          msg.ctx.digest_lines
                        } line(s)\n• Turns dropped: ${msg.ctx.dropped_turns}\n• Prompt ≈ ${msg.ctx.prompt_tokens} tokens`}
                      >
                        ⓘ context{msg.ctx.notes_injected ? " · notes ✓" : ""}
                        {msg.ctx.notes_truncated ? " (trimmed)" : ""}
                      </span>
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
            {/* While generating, the send button becomes Stop: aborts the stream
                and keeps whatever already arrived (marked "[answer stopped]"). */}
            <button
              className="send-icon"
              onClick={busy ? () => invoke("cancel_ask").catch(console.error) : send}
              disabled={!busy && (!collIds.length || !question.trim())}
              title={busy ? "Stop generating (keeps the partial answer)" : "Send (Enter)"}
              aria-label={busy ? "Stop" : "Send"}
            >
              {busy ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="5" y="5" width="14" height="14" rx="2" />
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="9 10 4 15 9 20" />
                  <path d="M20 4v7a4 4 0 0 1-4 4H4" />
                </svg>
              )}
            </button>
          </div>
          <div className="composer-bar">
            {/* Library selection lives at the top of the sidebar now. */}
            <div className="bar-left" />

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
        <div
          className={readerFull ? "reader full" : "reader"}
          ref={readerElRef}
          tabIndex={-1}
          onKeyDown={(e) => {
            if (e.key === "Escape" && readerFull) setReaderFull(false);
          }}
        >
          <div className="reader-head">
            <span className="name">
              {reader.path.split("/").pop()}
              {reader.page ? ` · p.${reader.page}` : ""}
            </span>
            <span className="reader-actions">
              <button
                className="ghost"
                onClick={() => setReaderFull((f) => !f)}
                title={
                  readerFull
                    ? reader.kind === "pdf" && reader.pdfNative
                      ? "Back to split view" // Esc dies once the native PDF iframe takes focus
                      : "Back to split view (Esc)"
                    : "Read full screen"
                }
              >
                {readerFull ? "⤡ Exit reader view" : "⛶ Reader view"}
              </button>
              <button className="ghost" onClick={() => setReader(null)}>
                ✕
              </button>
            </span>
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
          ) : reader.kind === "pdf" ? (
            reader.pdfNative ? (
              <iframe key={readerSrc} title="source" src={readerSrc} />
            ) : (
              <PdfReader
                key={reader.path}
                url={convertFileSrc(reader.path)}
                page={reader.page ?? undefined}
                full={readerFull}
                onFail={() =>
                  setReader((r) => (r && r.path === reader.path ? { ...r, pdfNative: true } : r))
                }
              />
            )
          ) : reader.kind === "md" ? (
            <div className="reader-md" ref={mdReaderRef}>
              {reader.error ? (
                <div className="reader-missing">
                  <h3>Couldn't preview this file</h3>
                  <p>{reader.error}</p>
                  <button className="primary" onClick={() => invoke("open_in_default_app", { path: reader.path }).catch(console.error)}>
                    Open in default app
                  </button>
                </div>
              ) : reader.text ? (
                <>
                  {reader.truncated && (
                    <div className="trunc-banner">
                      Showing the first 8 MB of{" "}
                      {((reader.totalBytes ?? 0) / (1024 * 1024)).toFixed(1)} MB.{" "}
                      <button
                        className="ghost"
                        onClick={() =>
                          invoke("open_in_default_app", { path: reader.path }).catch(console.error)
                        }
                      >
                        Open the full file in your default app
                      </button>
                    </div>
                  )}
                  {mdSliceOffsets
                    ? renderRich(reader.text.slice(0, mdSliceOffsets[mdSlices] ?? reader.text.length), [])
                    : renderRich(reader.text, [])}
                  {mdSliceOffsets && mdSlices < mdSliceOffsets.length - 1 && (
                    <div className="muted" style={{ padding: 8 }}>rendering…</div>
                  )}
                </>
              ) : (
                <div className="muted" style={{ padding: 16 }}>Loading…</div>
              )}
            </div>
          ) : (
            <div className="reader-missing">
              <h3>No in-app preview for this format yet</h3>
              <p>
                <code className="missing-path">{reader.path.split("/").pop()}</code> is an ebook format the
                built-in reader can't display (it renders PDFs and Markdown). The cited passage:
              </p>
              {reader.citeText && <blockquote className="cite-quote">{reader.citeText}</blockquote>}
              <button
                className="primary"
                onClick={() => invoke("open_in_default_app", { path: reader.path }).catch(console.error)}
                title="Open with the app your system associates with this format (e.g. Books or Calibre)"
              >
                Open in your ebook app
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
