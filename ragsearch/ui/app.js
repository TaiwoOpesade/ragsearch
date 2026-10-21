// ragsearch browser interface: a plain ES module with no build step and no dependencies.
// Text from documents, web pages and the model is always added with textContent or
// as DOM nodes, never parsed as markup, so an ingested page cannot inject script.

const $ = (id) => document.getElementById(id);

const ui = {
  status: $("status"), form: $("query-form"), q: $("q"), qLabel: $("q-label"),
  go: $("go"), stop: $("stop"), k: $("k"), web: $("web"), saveWeb: $("save-web"),
  webNote: $("web-note"), output: $("output"), announcer: $("announcer"),
  addOpen: $("add-open"), dialog: $("add-dialog"), addForm: $("add-form"),
  addCancel: $("add-cancel"), addSubmit: $("add-submit"), addResult: $("add-result"),
  filesFields: $("files-fields"), webFields: $("web-fields"), path: $("path"), url: $("url"),
  crawl: $("crawl"), crawlLimits: $("crawl-limits"), maxPages: $("max-pages"),
  maxDepth: $("max-depth"), tavilyProvider: $("tavily-provider"), tavilyCost: $("tavily-cost"),
};

const state = { health: null, controller: null, timer: null, addTimer: null };

const ERROR_TITLES = {
  0: "The ragsearch server isn't responding",
  409: "There's a problem with the index",
  422: "That didn't work",
  429: "The Tavily budget is used up",
  502: "Tavily couldn't help",
  503: "Ollama isn't available",
};

const UNKNOWN_ANSWER = /^i don['’]t know\b/i;
const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// --- small helpers -----------------------------------------------------------------

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false || child === "") continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const kbd = (text) => el("kbd", { text });
const plural = (n, one, many) => `${n.toLocaleString()} ${n === 1 ? one : many}`;
// Timings under 0.1 s only happen in test harnesses with a frozen clock, so leave them out.
const took = (seconds) => (seconds >= 0.1 ? ` in ${seconds.toFixed(1)} s` : "");
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const baseName = (path) => path.split(/[\\/]/).filter(Boolean).pop() || path;

// "C:/Users/me/notes/biology/cells.md" -> "…/notes/biology": enough to tell files apart.
function folderOf(path) {
  const parts = path.split(/[\\/]/).filter(Boolean).slice(0, -1);
  return parts.length > 2 ? `\u2026/${parts.slice(-2).join("/")}` : parts.join("/");
}

function clampInt(value, min, max, fallback) {
  const n = Number.parseInt(value, 10);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
}

function announce(message) {
  ui.announcer.textContent = "";
  requestAnimationFrame(() => { ui.announcer.textContent = message; });
}

function isTyping(target) {
  return target instanceof HTMLElement &&
    (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));
}

// --- talking to the API -------------------------------------------------------------

async function api(path, body, signal) {
  const options = body === undefined ? { signal } : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  };
  let response;
  try {
    response = await fetch(path, options);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(0, "Can't reach the ragsearch server. Is it still running?");
  }
  let data = null;
  try { data = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) throw new ApiError(response.status, detailOf(data) || `HTTP ${response.status}`);
  return data;
}

function detailOf(data) {
  const detail = data && data.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d && d.msg).filter(Boolean).join("; ");
  return "";
}

function errorNotice(err) {
  const title = ERROR_TITLES[err.status] || "Something went wrong";
  const hint = err.status === 503 ? " Start the Ollama app, then try again." : "";
  return el("div", { class: "notice notice-error", role: "alert" },
    el("strong", { text: title }), el("p", { text: err.message + hint }));
}

function warningsNotice(warnings) {
  if (!warnings || !warnings.length) return null;
  return el("div", { class: "notice", role: "status" },
    el("strong", { text: "Showing results from your notes only" }),
    warnings.map((w) => el("p", { text: w })));
}

// --- status bar and options -----------------------------------------------------------

function prettyEmbedder(name) {
  if (name.startsWith("hash-")) return "hash test embedder";
  return name.replace(/^ollama-/, "").replace(/\+prefix$/, "");
}

async function refreshHealth() {
  try {
    state.health = await api("/health");
  } catch (err) {
    ui.status.textContent = err.message;
    return null;
  }
  const h = state.health;
  const web = !h.web_configured ? "Web search off"
    : h.web_credits_used == null ? "Web search on"
    : `Web: ${h.web_credits_used} of ${h.web_credits_cap} credits used this month`;
  ui.status.replaceChildren(el("strong", { text: h.chunks.toLocaleString() }),
    ` ${h.chunks === 1 ? "chunk" : "chunks"} indexed · ${prettyEmbedder(h.embedder)} · ${web}`);
  updateWebOptions();
  return h;
}

function updateWebOptions() {
  const configured = Boolean(state.health && state.health.web_configured);
  ui.web.disabled = !configured;
  if (!configured) ui.web.checked = false;
  ui.saveWeb.disabled = !ui.web.checked;
  if (ui.saveWeb.disabled) ui.saveWeb.checked = false;
  ui.webNote.textContent = !configured
    ? "Web search is off. Add TAVILY_API_KEY to the .env file to turn it on."
    : ui.web.checked ? "The web is only searched when your notes don't match well. Each web search uses 1 Tavily credit." : "";
  const tavily = ui.tavilyProvider.querySelector("input");
  tavily.disabled = !configured;
  if (!configured && tavily.checked) ui.addForm.querySelector('input[value="local"]').checked = true;
  updateCost();
}

function mode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function applyMode() {
  const ask = mode() === "ask";
  ui.go.textContent = ask ? "Ask" : "Search";
  ui.q.placeholder = ask ? "Ask a question about your notes" : "Search your notes";
  ui.qLabel.textContent = ask ? "Your question" : "Search words";
}

// --- asking and searching -----------------------------------------------------------------

function showPending(message) {
  ui.go.disabled = true;
  ui.stop.hidden = false;
  // Only the message is in the live region; the ticking counter is not, so screen
  // readers announce the wait once instead of every second.
  const ticker = el("p", { class: "pending-time", "aria-hidden": "true" });
  ui.output.replaceChildren(el("div", { class: "pending", role: "status" },
    el("div", { class: "pending-bar" }), el("p", { text: message })), ticker);
  const started = Date.now();
  clearInterval(state.timer);
  state.timer = setInterval(() => {
    const seconds = Math.round((Date.now() - started) / 1000);
    if (seconds >= 4) {
      ticker.textContent = `${seconds} s so far. The first question after Ollama starts ` +
        "is slower while the model loads.";
    }
  }, 1000);
}

function endPending() {
  clearInterval(state.timer);
  ui.go.disabled = false;
  ui.stop.hidden = true;
}

async function run(event) {
  if (event) event.preventDefault();
  const text = ui.q.value.trim();
  if (!text) { ui.q.focus(); return; }
  if (state.controller) state.controller.abort();
  const controller = new AbortController();
  state.controller = controller;
  const ask = mode() === "ask";
  const body = {
    ...(ask ? { question: text } : { query: text }),
    k: Number(ui.k.value), web: ui.web.checked, save_web: ui.saveWeb.checked,
  };
  rememberQuery(text);
  showPending(ask ? "Reading your notes and writing an answer." : "Searching your notes.");
  const started = performance.now();
  try {
    const data = await api(ask ? "/ask" : "/search", body, controller.signal);
    const seconds = (performance.now() - started) / 1000;
    if (ask) renderAnswer(data, seconds); else renderSearch(text, data, seconds);
    if (body.web || body.save_web) refreshHealth();
  } catch (err) {
    if (err.name === "AbortError") {
      if (state.controller === controller) {
        ui.output.replaceChildren(el("p", { class: "empty-results", text: "Stopped waiting. " +
          "The server may still finish this question in the background." }));
      }
    } else {
      ui.output.replaceChildren(errorNotice(err));
    }
  } finally {
    if (state.controller === controller) {
      state.controller = null;
      endPending();
    }
  }
}

function withCitations(text, count) {
  const parts = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    parts.push(text.slice(last, match.index));
    // Same rule as the server: in-range numbers are citations, others stay as text.
    for (const n of match[1].split(",").map((x) => Number(x.trim()))) {
      parts.push(n >= 1 && n <= count ? citeLink(n) : `[${n}]`);
    }
    last = match.index + match[0].length;
  }
  parts.push(text.slice(last));
  return parts;
}

function citeLink(n) {
  return el("a", {
    class: "cite", href: `#source-${n}`, "aria-label": `Source ${n}`,
    onclick: (event) => { event.preventDefault(); focusSource(n); },
  }, String(n));
}

function focusSource(n) {
  const target = document.getElementById(`source-${n}`);
  if (!target) return;
  target.classList.remove("flash");
  void target.offsetWidth; // restart the highlight animation
  target.classList.add("flash");
  target.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "nearest" });
  target.focus({ preventScroll: true });
}

function renderAnswer(data, seconds) {
  const cited = new Set(data.citations.map((c) => c.number));
  const unknown = data.citations.length === 0 && UNKNOWN_ANSWER.test(data.answer.trim());
  const answer = unknown ? unknownAnswer(data) : el("article", { class: "answer", "aria-label": "Answer" },
    el("p", { class: "answer-text" }, withCitations(data.answer, data.results.length)),
    el("p", { class: "answer-meta",
      text: `Answered from ${plural(data.results.length, "source", "sources")}${took(seconds)}` }));
  ui.output.replaceChildren(...[warningsNotice(data.warnings), answer,
    data.results.length ? sourceList("Sources", "", data.results, cited) : null].filter(Boolean));
  announce(unknown ? "Your notes don't cover this question."
    : `Answer ready. It cites ${plural(cited.size, "source", "sources")}.`);
}

function unknownAnswer(data) {
  const configured = Boolean(state.health && state.health.web_configured);
  // The web is only searched when the best local match is below the threshold, so a
  // retry would change nothing when the notes matched well but still lacked the answer.
  const best = data.results.reduce((top, r) => Math.max(top, r.score), -1);
  const threshold = state.health ? state.health.web_fallback_threshold : 0.55;
  const webWouldRun = best < threshold;
  const canUseWeb = configured && !ui.web.checked && webWouldRun;
  const hint = !configured
    ? "Try rephrasing or add more notes. To search the web too, add TAVILY_API_KEY to the .env file."
    : webWouldRun ? "Try rephrasing, add more notes, or let ragsearch search the web."
    : "Your notes matched this question closely but don't answer it. Try rephrasing or add more notes.";
  return el("article", { class: "answer answer-unknown", "aria-label": "Answer" },
    el("h2", { text: "Your notes don't cover this" }),
    el("p", { class: "answer-text", text: data.answer }),
    el("p", { class: "answer-meta", text: hint }),
    canUseWeb ? el("div", { class: "answer-actions" }, el("button", {
      type: "button", class: "btn",
      onclick: () => { ui.web.checked = true; updateWebOptions(); run(); },
    }, "Try again with web search (1 credit)")) : null);
}

function renderSearch(query, data, seconds) {
  if (!data.results.length) {
    const empty = state.health && state.health.chunks === 0
      ? "Your index is empty. Add some documents first." : "Nothing in your index matched.";
    ui.output.replaceChildren(...[warningsNotice(data.warnings),
      el("p", { class: "empty-results", text: empty })].filter(Boolean));
    announce(empty);
    return;
  }
  const extra = `for “${query}”${seconds >= 0.1 ? ` · ${seconds.toFixed(1)} s` : ""}`;
  ui.output.replaceChildren(...[warningsNotice(data.warnings),
    sourceList("Results", extra, data.results, new Set())].filter(Boolean));
  announce(`${plural(data.results.length, "result", "results")} found.`);
}

function sourceList(title, extra, results, cited) {
  return el("section", { "aria-label": title },
    el("h2", { class: "results-title" }, title, extra ? el("span", { text: ` ${extra}` }) : null),
    el("ol", { class: "source-list" },
      results.map((r, i) => sourceItem(r, r.number || i + 1, cited.has(r.number || i + 1)))));
}

function sourceItem(result, n, cited) {
  const live = result.origin === "web"; // fetched from the web for this question
  const web = live || /^https?:\/\//i.test(result.source); // any web page, even if saved
  const snippet = el("p", { class: "snippet clamped", text: result.text });
  const tools = el("div", { class: "source-tools" });
  if (!web && navigator.clipboard) tools.append(copyButton(result.source));
  const item = el("li", { class: cited ? "source cited" : "source", id: `source-${n}`, tabindex: "-1" },
    el("span", { class: "num", "aria-hidden": "true", text: String(n) }),
    el("div", { class: "source-body" },
      el("div", { class: "source-head" },
        el("span", { class: live ? "badge badge-web" : "badge",
          text: live ? "Web" : web ? "Saved page" : "Your notes" }),
        web ? webLink(result.source) : el("span", { class: "source-name", text: baseName(result.source) }),
        cited ? el("span", { class: "visually-hidden", text: "(cited in the answer)" }) : null,
        el("span", { class: "score", title: "Cosine similarity to your question (1 = identical)",
          text: `${result.score.toFixed(2)} match` })),
      web ? null : el("p", { class: "source-path", title: result.source, text: folderOf(result.source) }),
      snippet, tools));
  requestAnimationFrame(() => {
    if (snippet.scrollHeight > snippet.clientHeight + 2) tools.prepend(expandButton(snippet));
  });
  return item;
}

function webLink(url) {
  let parsed = null;
  try { parsed = new URL(url); } catch { /* not a URL */ }
  if (!parsed || !["http:", "https:"].includes(parsed.protocol)) {
    return el("span", { class: "source-name", text: url });
  }
  let path = parsed.pathname === "/" ? "" : parsed.pathname;
  try { path = decodeURI(path); } catch { /* keep it encoded */ }
  return el("a", { class: "source-name", href: parsed.href, target: "_blank",
    rel: "noopener noreferrer", text: parsed.hostname.replace(/^www\./, "") + path });
}

function expandButton(snippet) {
  const button = el("button", { type: "button", class: "link-btn", "aria-expanded": "false" }, "Show more");
  button.addEventListener("click", () => {
    const open = snippet.classList.toggle("clamped") === false;
    button.textContent = open ? "Show less" : "Show more";
    button.setAttribute("aria-expanded", String(open));
  });
  return button;
}

function copyButton(path) {
  const button = el("button", { type: "button", class: "link-btn" }, "Copy path");
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(path);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Couldn't copy";
    }
    setTimeout(() => { button.textContent = "Copy path"; }, 1600);
  });
  return button;
}

function renderIdle() {
  if (!state.health) return;
  if (state.health.chunks === 0) {
    ui.output.replaceChildren(el("div", { class: "welcome" },
      el("h2", { text: "Start by adding your notes" }),
      el("p", { text: "ragsearch reads .md, .txt and .pdf files from a folder and indexes them " +
        "on this computer. Nothing leaves it unless you turn on web search." }),
      el("button", { type: "button", class: "btn btn-primary", onclick: openAdd }, "Add a folder")));
  } else {
    ui.output.replaceChildren(el("p", { class: "tips" }, "Press ", kbd("/"), " to start typing, ",
      kbd("Enter"), " to send and ", kbd("Shift"), " + ", kbd("Enter"), " for a new line."));
  }
}

function rememberQuery(text) {
  const url = new URL(window.location.href);
  url.searchParams.set("q", text);
  url.searchParams.set("mode", mode());
  window.history.replaceState(null, "", url);
}

// --- adding documents ------------------------------------------------------------------------

function kind() {
  return document.querySelector('input[name="kind"]:checked').value;
}

function provider() {
  return document.querySelector('input[name="provider"]:checked').value;
}

function updateKind() {
  const files = kind() === "files";
  ui.filesFields.hidden = !files;
  ui.webFields.hidden = files;
  ui.crawlLimits.hidden = !ui.crawl.checked;
  updateCost();
}

function updateCost() {
  if (!state.health || !state.health.web_configured) {
    ui.tavilyCost.textContent = "Needs TAVILY_API_KEY in the .env file.";
    return;
  }
  const pages = clampInt(ui.maxPages.value, 1, 500, 30);
  const credits = ui.crawl.checked ? Math.ceil(pages / 10) + Math.ceil(pages / 5) : 1;
  ui.tavilyCost.textContent =
    `Faster. Uses ${ui.crawl.checked ? "up to " : ""}${plural(credits, "Tavily credit", "Tavily credits")}.`;
}

function openAdd() {
  if (!state.addTimer) ui.addResult.replaceChildren();
  updateKind();
  ui.dialog.showModal();
  (kind() === "files" ? ui.path : ui.url).focus();
}

function fieldProblem(input, message) {
  input.setAttribute("aria-invalid", "true");
  input.focus();
  ui.addResult.replaceChildren(el("p", { class: "notice notice-error", role: "alert", text: message }));
}

function setAddBusy(busy) {
  ui.addSubmit.disabled = busy;
  ui.addSubmit.textContent = busy ? "Adding…" : "Add";
  clearInterval(state.addTimer);
  state.addTimer = null;
  if (!busy) return;
  const started = Date.now();
  const ticker = el("p", { class: "pending-time", "aria-hidden": "true" });
  ui.addResult.replaceChildren(el("div", { class: "pending" }, el("div", { class: "pending-bar" }),
    el("p", { text: "Working. Big folders and site crawls take a while; you can close this " +
      "window and keep using ragsearch." }), ticker));
  state.addTimer = setInterval(() => {
    ticker.textContent = `${Math.round((Date.now() - started) / 1000)} s so far.`;
  }, 1000);
}

async function submitAdd(event) {
  event.preventDefault();
  for (const input of [ui.path, ui.url]) input.removeAttribute("aria-invalid");
  let body;
  if (kind() === "files") {
    const path = ui.path.value.trim().replace(/^"(.*)"$/, "$1"); // Explorer's "Copy as path"
    if (!path) return fieldProblem(ui.path, "Enter the path of a folder or file.");
    body = { path };
  } else {
    let url = ui.url.value.trim();
    if (!url) return fieldProblem(ui.url, "Enter a web address.");
    if (!/^https?:\/\//i.test(url)) url = `https://${url}`;
    url = url.replace(/^https?/i, (scheme) => scheme.toLowerCase()).replace(/ /g, "%20");
    body = {
      url, crawl: ui.crawl.checked, provider: provider(),
      max_pages: clampInt(ui.maxPages.value, 1, 500, 30),
      max_depth: clampInt(ui.maxDepth.value, 0, 5, 2),
    };
  }
  setAddBusy(true);
  try {
    const report = await api("/ingest", body);
    setAddBusy(false);
    renderReport(report);
    await refreshHealth();
    const idle = !ui.output.firstChild || ui.output.querySelector(".welcome, .tips");
    if (!state.controller && idle) renderIdle();
  } catch (err) {
    setAddBusy(false);
    ui.addResult.replaceChildren(errorNotice(err));
  }
}

function renderReport(report) {
  const sources = report.updated.length + report.skipped.length + report.failed.length;
  const details = [`${report.skipped.length} unchanged`, `${report.failed.length} failed`];
  if (report.removed.length) details.push(`${report.removed.length} removed`);
  const failed = report.failed.slice(0, 8).map((s) => el("li", { text: baseName(s) }));
  if (report.failed.length > 8) failed.push(el("li", { text: `and ${report.failed.length - 8} more` }));
  ui.addResult.replaceChildren(
    el("p", { class: "summary",
      text: `Added ${plural(report.added, "new chunk", "new chunks")} from ${plural(sources, "source", "sources")}.` }),
    el("p", { class: "hint", text: `${details.join(", ")}. The index now holds ${plural(report.total_chunks, "chunk", "chunks")}.` }),
    failed.length ? el("ul", { "aria-label": "Files that could not be read" }, failed) : null);
  announce(`Added ${plural(report.added, "new chunk", "new chunks")}.`);
}

// --- wiring --------------------------------------------------------------------------------

function wire() {
  ui.form.addEventListener("submit", run);
  ui.stop.addEventListener("click", () => { if (state.controller) state.controller.abort(); });
  ui.q.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      ui.form.requestSubmit();
    }
  });
  document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", applyMode));
  ui.web.addEventListener("change", updateWebOptions);
  document.addEventListener("keydown", (event) => {
    if (ui.dialog.open) return;
    if (event.key === "/" && !isTyping(event.target)) {
      event.preventDefault();
      ui.q.focus();
    } else if (event.key === "Escape" && state.controller) {
      state.controller.abort();
    }
  });

  ui.addOpen.addEventListener("click", openAdd);
  ui.addCancel.addEventListener("click", () => ui.dialog.close());
  ui.dialog.addEventListener("click", (event) => { if (event.target === ui.dialog) ui.dialog.close(); });
  ui.addForm.addEventListener("submit", submitAdd);
  document.querySelectorAll('input[name="kind"]').forEach((r) => r.addEventListener("change", updateKind));
  ui.crawl.addEventListener("change", updateKind);
  ui.maxPages.addEventListener("input", updateCost);
}

async function init() {
  wire();
  const params = new URLSearchParams(window.location.search);
  if (params.get("mode") === "search") $("mode-search").checked = true;
  applyMode();
  const query = params.get("q");
  if (query) ui.q.value = query;
  const health = await refreshHealth();
  if (window.location.hash === "#add") openAdd();
  // A shared link may fill in and run a query, but never with web search on.
  if (query && health && health.chunks > 0) run();
  else renderIdle();
}

init();
