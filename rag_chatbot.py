"""Fall Armyworm RAG chatbot with a knowledge base YOU control.

Edited from: Chatbot and RAG.ipynb
Add your own sources (PDF, text/Markdown, web pages, pasted text) and the chatbot
retrieves from them when answering, citing each source as [1], [2], ...

Ways to add sources
    1. Streamlit page     -> upload files, paste a web link, or paste text (see render_rag_tab)
    2. Drop-in folder     -> copy files into knowledge_base/inbox/ (loaded automatically at start-up)
    3. Python             -> bot.add_file(...), bot.add_url(...), bot.add_text(...)
    4. Command line       -> python rag_chatbot.py add-file guide.pdf   (see --help)

Run (terminal):   python rag_chatbot.py demo
                  python rag_chatbot.py ask "How do I scout for fall armyworm?"
Run (Streamlit):  streamlit run rag_chatbot.py
Reuse in your app:
    from rag_chatbot import render_rag_tab
    with tab_chat:
        render_rag_tab(lang)              # lang = "en" or "sw"

Answers need an Anthropic key (ANTHROPIC_API_KEY, env var or Streamlit secret). Without one
the chatbot still works in retrieval-only mode and shows the matching passages.
Optional: ADMIN_PASSWORD (env/secret) locks the add/remove-source controls on a public app.

Limits to know about
  * Retrieval is TF-IDF (matches words, not meaning). When a key is set, the question is first
    rewritten into English + Kiswahili keywords so Kiswahili questions can find English sources.
  * Scanned PDFs (images of pages) have no text and are rejected: run OCR on them first.
  * On Streamlit Community Cloud the disk is wiped on reboot: sources added through the web
    page disappear. To keep them, commit the files to knowledge_base/inbox/ in your repo.
Dependencies: scikit-learn numpy requests pypdf streamlit anthropic
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_KB_DIR = "knowledge_base"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm"}
MAX_BYTES = 10 * 1024 * 1024          # per file / web page
CHUNK_WORDS, CHUNK_OVERLAP = 150, 30
DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"   # override with CLAUDE_MODEL

# Starter sources from the original notebook (used by `demo`).
SAMPLE_SOURCES = [
    {"title": "Chemical Treatment for Fall Armyworm",
     "content": "Apply registered chemical insecticides like Emamectin benzoate or Spinetoram when more than 20% of young maize plants show damage symptoms."},
    {"title": "Cultural Control and Sowing Advice",
     "content": "Early sowing is highly recommended for maize. Late planted maize is often heavily infested by migrating populations of Fall Armyworm moths."},
    {"title": "Physical Prevention Methods",
     "content": "Smallholder farmers can handpick and destroy egg masses and young caterpillars. Dropping clean soil, sand, or wood ash into maize whorls can also help crush or suffocate larvae."},
]

SYSTEM_PROMPT = """You are an agronomy assistant helping small-scale maize farmers in Kenya manage fall armyworm.
Rules:
- Reply in the same language as the farmer's latest message (English or Kiswahili), in simple, clear words.
- Use ONLY the numbered SOURCES provided. Put the source number in brackets, like [1], after each claim.
- If the sources do not answer the question, say so plainly and suggest the county agricultural extension officer. Do not guess.
- Never invent chemical names, doses, spray intervals or prices that are not in the sources. For any pesticide, remind the farmer to use only products registered in Kenya (check with PCPB or an extension officer) and to follow the label.
- Keep answers short and practical."""

REWRITE_PROMPT = ("Turn the farmer's message into search keywords about maize pests. Output ONE line: "
                  "the keywords in English followed by the same keywords in Kiswahili. No explanations.")


# --------------------------------------------------------------------------- text extraction
def _clean(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "form", "svg"}
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article",
             "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title, self._skip, self._in_title = [], "", 0, False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def extract_sections(data: bytes, filename: str) -> tuple[str, list[dict]]:
    """Return (suggested title, [{'label': 'page 3', 'text': ...}, ...]) for a file's bytes."""
    ext = Path(filename).suffix.lower()
    title = ""
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("The PDF is password-protected.")
        title = (reader.metadata.title or "") if reader.metadata else ""
        sections = [{"label": f"page {i}", "text": _clean(p.extract_text() or "")}
                    for i, p in enumerate(reader.pages, 1)]
    elif ext in {".html", ".htm"}:
        parser = _HTMLText()
        parser.feed(data.decode("utf-8", errors="replace"))
        title = parser.title.strip()
        sections = [{"label": "", "text": _clean("".join(parser.parts))}]
    elif ext in {".txt", ".md"}:
        sections = [{"label": "", "text": _clean(data.decode("utf-8", errors="replace"))}]
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    sections = [s for s in sections if s["text"]]
    if not sections:
        raise ValueError("No readable text found (is this a scanned PDF? Run OCR on it first).")
    return title.strip(), sections


def _chunks(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    step = max(size - overlap, 1)
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + size])
        if piece:
            yield piece
        if i + size >= len(words):
            break


def _hash(*parts: bytes | str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode("utf-8"))
    return h.hexdigest()[:10]


# --------------------------------------------------------------------------- the chatbot
class MaizeRAGChatbot:
    """Knowledge base + TF-IDF retriever + prompt builder (+ optional Claude answers)."""

    def __init__(self, kb_dir: str | Path = DEFAULT_KB_DIR, auto_sync_inbox: bool = True):
        self.root = Path(kb_dir)
        self.texts_dir = self.root / "texts"
        self.inbox_dir = self.root / "inbox"
        self.registry_path = self.root / "sources.json"
        for d in (self.texts_dir, self.inbox_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()
        self._chunks: list[dict] = []
        self._vectorizer = None
        self._matrix = None
        self._dirty = True
        self.sync_errors: list[str] = []
        if auto_sync_inbox:
            self.sync_inbox()

    # ---- registry -----------------------------------------------------------------
    def _load_registry(self) -> dict:
        if self.registry_path.exists():
            try:
                reg = json.loads(self.registry_path.read_text(encoding="utf-8"))
                reg.setdefault("sources", [])
                reg.setdefault("removed", [])
                return reg
            except json.JSONDecodeError:
                pass
        return {"sources": [], "removed": []}

    def _save_registry(self) -> None:
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._registry, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.registry_path)

    def _register(self, source_id: str, title: str, kind: str, origin: str, sections: list[dict],
                  explicit_title: bool = False) -> dict:
        if not explicit_title:   # re-adding the same source must not overwrite a title you chose earlier
            title = next((s["title"] for s in self._registry["sources"] if s["id"] == source_id), title)
        (self.texts_dir / f"{source_id}.json").write_text(
            json.dumps(sections, ensure_ascii=False), encoding="utf-8")
        entry = {"id": source_id, "title": title or origin, "type": kind, "origin": origin,
                 "added": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "words": sum(len(s["text"].split()) for s in sections)}
        self._registry["sources"] = [s for s in self._registry["sources"] if s["id"] != source_id] + [entry]
        if source_id in self._registry["removed"]:
            self._registry["removed"].remove(source_id)
        self._save_registry()
        self._dirty = True
        return entry

    # ---- adding / removing sources -------------------------------------------------
    def add_file(self, source: str | Path | bytes, filename: str | None = None,
                 title: str | None = None) -> dict:
        """Add a .pdf/.txt/.md/.html file from a path or from raw bytes (+ filename)."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            data, filename = path.read_bytes(), filename or path.name
        else:
            data = source
            if not filename:
                raise ValueError("filename is required when passing bytes.")
        if len(data) > MAX_BYTES:
            raise ValueError(f"File is larger than {MAX_BYTES // 1_000_000} MB.")
        suggested, sections = extract_sections(data, filename)
        return self._register(_hash(data), title or suggested or Path(filename).stem.replace("_", " "),
                              "file", filename, sections, explicit_title=bool(title))

    def add_url(self, url: str, title: str | None = None) -> dict:
        """Download a web page or PDF and add it. Re-adding the same URL refreshes it."""
        url = url.strip()
        if urlparse(url).scheme not in {"http", "https"}:
            raise ValueError("The link must start with http:// or https://")
        r = requests.get(url, timeout=20, headers={"User-Agent": "FarmerAssistant/1.0"})
        r.raise_for_status()
        if len(r.content) > MAX_BYTES:
            raise ValueError(f"Page is larger than {MAX_BYTES // 1_000_000} MB.")
        is_pdf = "pdf" in r.headers.get("content-type", "").lower() or urlparse(url).path.lower().endswith(".pdf")
        suggested, sections = extract_sections(r.content, "page.pdf" if is_pdf else "page.html")
        return self._register(_hash(url), title or suggested or url, "url", url, sections,
                              explicit_title=bool(title))

    def add_text(self, title: str, text: str) -> dict:
        """Add pasted text (e.g. an extension officer's notes) as a source."""
        text = _clean(text)
        if not title.strip() or not text:
            raise ValueError("Both a title and some text are required.")
        return self._register(_hash(title, text), title.strip(), "text", "pasted text",
                              [{"label": "", "text": text}], explicit_title=True)

    def load_documents(self, doc_list: list[dict]) -> None:
        """Backward-compatible with the notebook: [{'title':..., 'content':...}, ...]."""
        for d in doc_list:
            self.add_text(d["title"], d["content"])
        print(f"Successfully loaded and indexed {len(doc_list)} documents.")

    def remove_source(self, source_id: str) -> bool:
        before = len(self._registry["sources"])
        self._registry["sources"] = [s for s in self._registry["sources"] if s["id"] != source_id]
        if len(self._registry["sources"]) == before:
            return False
        (self.texts_dir / f"{source_id}.json").unlink(missing_ok=True)
        if source_id not in self._registry["removed"]:
            self._registry["removed"].append(source_id)   # stops inbox files re-appearing
        self._save_registry()
        self._dirty = True
        return True

    def list_sources(self) -> list[dict]:
        return list(self._registry["sources"])

    def sync_inbox(self) -> int:
        """Load any new files from knowledge_base/inbox/. Returns how many were added."""
        added, self.sync_errors = 0, []
        known = {s["id"] for s in self._registry["sources"]} | set(self._registry["removed"])
        for path in sorted(self.inbox_dir.glob("**/*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                if _hash(path.read_bytes()) in known:
                    continue
                self.add_file(path)
                added += 1
            except Exception as exc:  # keep going; report at the end
                self.sync_errors.append(f"{path.name}: {exc}")
        return added

    # ---- retrieval -----------------------------------------------------------------
    def _build_index(self) -> None:
        self._chunks = []
        for src in self._registry["sources"]:
            path = self.texts_dir / f"{src['id']}.json"
            if not path.exists():
                continue
            for sec in json.loads(path.read_text(encoding="utf-8")):
                for piece in _chunks(sec["text"]):
                    self._chunks.append({"source_id": src["id"], "title": src["title"],
                                         "origin": src["origin"], "label": sec["label"], "content": piece})
        if self._chunks:
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")
            self._matrix = self._vectorizer.fit_transform([c["content"] for c in self._chunks])
        else:
            self._vectorizer = self._matrix = None
        self._dirty = False

    def retrieve(self, query: str, top_k: int = 4, threshold: float = 0.05, max_per_source: int = 2) -> list[dict]:
        """Most relevant passages, with at most `max_per_source` from any one source."""
        if self._dirty:
            self._build_index()
        if self._matrix is None or not query.strip():
            return []
        sims = cosine_similarity(self._vectorizer.transform([query]), self._matrix).flatten()
        results, per_source, seen = [], {}, set()
        for idx in np.argsort(sims)[::-1]:
            if sims[idx] < threshold or len(results) >= top_k:
                break
            c = self._chunks[idx]
            if per_source.get(c["source_id"], 0) >= max_per_source or c["content"] in seen:
                continue
            seen.add(c["content"])
            per_source[c["source_id"]] = per_source.get(c["source_id"], 0) + 1
            results.append({**c, "score": float(sims[idx])})
        return results

    # ---- prompt + answer -----------------------------------------------------------
    def get_chat_response(self, user_message: str, chat_history=None, search_query: str | None = None) -> dict:
        """Retrieve passages and build the prompt (what the notebook version did), now with citations."""
        docs = self.retrieve(search_query or user_message)
        citations = [{"n": i, "title": d["title"], "label": d["label"], "origin": d["origin"]}
                     for i, d in enumerate(docs, 1)]
        if docs:
            context = "\n\n".join(
                f"[{c['n']}] {c['title']}{' (' + c['label'] + ')' if c['label'] else ''}\n{d['content']}"
                for c, d in zip(citations, docs))
        else:
            context = "No relevant sources were found in the knowledge base."
        prompt = (f"{SYSTEM_PROMPT}\n\nSOURCES:\n{context}\n\nFarmer's question:\n{user_message}\n\nAnswer:")
        return {"prompt_to_llm": prompt, "retrieved_documents": docs, "citations": citations, "context": context}

    def _client(self):
        key = _secret("ANTHROPIC_API_KEY")
        if not key:
            return None
        import anthropic
        return anthropic.Anthropic(api_key=key)

    def _llm(self, system: str, messages: list[dict], max_tokens: int = 700) -> str:
        r = self._client().messages.create(model=_secret("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
                                           max_tokens=max_tokens, system=system, messages=messages)
        return "".join(b.text for b in r.content if b.type == "text")

    def answer(self, user_message: str, chat_history: list[dict] | None = None) -> dict:
        """Full RAG turn. 'answer' is None when no API key is set (retrieval-only mode)."""
        if self._client() is None:
            payload = self.get_chat_response(user_message)
            return {**payload, "answer": None}
        keywords = self._llm(REWRITE_PROMPT, [{"role": "user", "content": user_message}], max_tokens=80)
        payload = self.get_chat_response(user_message, search_query=f"{keywords} {user_message}")
        past = (chat_history or [])[-6:]
        while past and past[0]["role"] != "user":
            past = past[1:]
        msgs = past + [{"role": "user", "content": f"SOURCES:\n{payload['context']}\n\nQuestion: {user_message}"}]
        return {**payload, "answer": self._llm(SYSTEM_PROMPT, msgs)}


# --------------------------------------------------------------------------- secrets / env
def _secret(name: str, default=None):
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def _make_bot() -> MaizeRAGChatbot:
    return MaizeRAGChatbot()


if _in_streamlit():                      # one shared bot per server process
    import streamlit as _st
    _make_bot = _st.cache_resource(_make_bot)
else:
    _make_bot = functools.lru_cache(maxsize=1)(_make_bot)


def get_bot() -> MaizeRAGChatbot:
    """The shared chatbot (added sources stay visible across Streamlit reruns)."""
    return _make_bot()


# --------------------------------------------------------------------------- Streamlit UI
LABELS = {
    "en": {"title": "🌽 Fall armyworm advisor", "sources": "Knowledge sources", "none": "No sources yet. Add some above so the chatbot has something to cite.",
           "locked": "Admin password", "wrong": "Wrong password.", "up": "Upload files (PDF, TXT, MD, HTML)", "add": "Add",
           "url": "Web page or PDF link", "url_title": "Title (optional)", "paste_title": "Title", "paste": "Paste text",
           "tabs": ["Files", "Web link", "Paste text"], "remove": "Remove", "added": "Added: {t}", "failed": "Could not add {n}: {e}",
           "ask": "Ask about fall armyworm...", "src": "Sources", "nokey": "No API key set: showing the closest passages instead of a written answer.",
           "nomatch": "Nothing relevant was found in the sources. Try adding more sources or rephrasing.",
           "err": "Something went wrong. Please try again.", "meta": "{k} · {w} words · added {d}"},
    "sw": {"title": "🌽 Mshauri wa viwavi jeshi vamizi", "sources": "Vyanzo vya maarifa", "none": "Bado hakuna vyanzo. Ongeza hapo juu ili chatbot iwe na cha kunukuu.",
           "locked": "Nenosiri la msimamizi", "wrong": "Nenosiri si sahihi.", "up": "Pakia faili (PDF, TXT, MD, HTML)", "add": "Ongeza",
           "url": "Kiungo cha ukurasa wa wavuti au PDF", "url_title": "Kichwa (si lazima)", "paste_title": "Kichwa", "paste": "Bandika maandishi",
           "tabs": ["Faili", "Kiungo cha wavuti", "Bandika maandishi"], "remove": "Ondoa", "added": "Imeongezwa: {t}", "failed": "Imeshindwa kuongeza {n}: {e}",
           "ask": "Uliza kuhusu viwavi jeshi vamizi...", "src": "Vyanzo", "nokey": "Ufunguo wa API haujawekwa: inaonyesha vifungu vilivyo karibu zaidi badala ya jibu.",
           "nomatch": "Hakuna kilichopatikana kwenye vyanzo. Jaribu kuongeza vyanzo zaidi au uliza kwa maneno mengine.",
           "err": "Hitilafu imetokea. Tafadhali jaribu tena.", "meta": "{k} · maneno {w} · iliongezwa {d}"},
}


def _cite_line(c: dict, numbered: bool = True) -> str:
    where = f", {c['label']}" if c["label"] else ""
    origin = f" — {c['origin']}" if c["origin"] not in {"pasted text", c["title"]} else ""
    return f"{'[%d] ' % c['n'] if numbered else ''}{c['title']}{where}{origin}"


def render_rag_tab(lang: str = "en", key_prefix: str = "rag") -> None:
    """Source manager + chat. Call inside a tab/page of your Streamlit app."""
    import streamlit as st
    t, bot, p = LABELS[lang], get_bot(), key_prefix
    st.session_state.setdefault(f"{p}_msgs", [])
    st.session_state.setdefault(f"{p}_n", 0)

    with st.expander(f"{t['sources']} ({len(bot.list_sources())})", expanded=not bot.list_sources()):
        admin_pw = _secret("ADMIN_PASSWORD")
        unlocked = True
        if admin_pw:
            unlocked = st.text_input(t["locked"], type="password", key=f"{p}_pw") == admin_pw
            if st.session_state.get(f"{p}_pw") and not unlocked:
                st.error(t["wrong"])
        if unlocked:
            tab_f, tab_u, tab_t = st.tabs(t["tabs"])
            n = st.session_state[f"{p}_n"]           # changing the key resets the widgets after adding
            with tab_f:
                files = st.file_uploader(t["up"], type=[e.strip(".") for e in sorted(SUPPORTED_EXTENSIONS)],
                                         accept_multiple_files=True, key=f"{p}_up{n}")
                if files and st.button(t["add"], key=f"{p}_addf{n}"):
                    for f in files:
                        try:
                            st.toast(t["added"].format(t=bot.add_file(f.getvalue(), f.name)["title"]))
                        except Exception as exc:
                            st.error(t["failed"].format(n=f.name, e=exc))
                    st.session_state[f"{p}_n"] += 1
                    st.rerun()
            with tab_u:
                url = st.text_input(t["url"], key=f"{p}_url{n}")
                url_title = st.text_input(t["url_title"], key=f"{p}_ut{n}")
                if url and st.button(t["add"], key=f"{p}_addu{n}"):
                    try:
                        st.toast(t["added"].format(t=bot.add_url(url, url_title or None)["title"]))
                        st.session_state[f"{p}_n"] += 1
                        st.rerun()
                    except Exception as exc:
                        st.error(t["failed"].format(n=url, e=exc))
            with tab_t:
                pt = st.text_input(t["paste_title"], key=f"{p}_pt{n}")
                body = st.text_area(t["paste"], height=150, key=f"{p}_pb{n}")
                if st.button(t["add"], key=f"{p}_addt{n}"):
                    try:
                        st.toast(t["added"].format(t=bot.add_text(pt, body)["title"]))
                        st.session_state[f"{p}_n"] += 1
                        st.rerun()
                    except Exception as exc:
                        st.error(t["failed"].format(n="text", e=exc))

        for e in bot.sync_errors:
            st.warning(e)
        if not bot.list_sources():
            st.info(t["none"])
        for s in bot.list_sources():
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{s['title']}**  \n<small>{t['meta'].format(k=s['type'], w=s['words'], d=s['added'][:10])}</small>",
                        unsafe_allow_html=True)
            if unlocked and c2.button(t["remove"], key=f"{p}_rm_{s['id']}"):
                bot.remove_source(s["id"])
                st.rerun()

    for m in st.session_state[f"{p}_msgs"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("cites"):
                st.caption(t["src"] + ": " + " | ".join(m["cites"]))

    q = st.chat_input(t["ask"], key=f"{p}_input")
    if not q:
        return
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("..."):
            try:
                res = bot.answer(q, st.session_state[f"{p}_msgs"])
            except Exception:
                st.error(t["err"])
                return
        if res["answer"] is None:
            if bot.list_sources():
                st.caption(t["nokey"])
            text = "\n\n".join(f"**[{c['n']}]** {d['content'][:500]}..."
                               for c, d in zip(res["citations"], res["retrieved_documents"])) or t["nomatch"]
        else:
            text = res["answer"]
        st.markdown(text)
        cites = [_cite_line(c) for c in res["citations"]]
        if cites:
            st.caption(t["src"] + ": " + " | ".join(cites))
    st.session_state[f"{p}_msgs"] += [{"role": "user", "content": q},
                                      {"role": "assistant", "content": text, "cites": cites}]


def _streamlit_entry() -> None:
    import streamlit as st
    st.set_page_config(page_title="Fall armyworm advisor", page_icon="🌽")
    lang = st.sidebar.radio("Language / Lugha", ["en", "sw"], format_func={"en": "English", "sw": "Kiswahili"}.get)
    st.title(LABELS[lang]["title"])
    render_rag_tab(lang)


# --------------------------------------------------------------------------- CLI
def _print_result(res: dict) -> None:
    if res.get("answer"):
        print(res["answer"])
    elif res["retrieved_documents"]:
        print("(retrieval-only mode: no ANTHROPIC_API_KEY set)")
        for c, d in zip(res["citations"], res["retrieved_documents"]):
            print(f"\n[{c['n']}] {_cite_line(c, False)}  (score {d['score']:.2f})\n    {d['content'][:300]}")
    else:
        print("No relevant sources found.")
    if res["citations"]:
        print("\nSources:\n  " + "\n  ".join(_cite_line(c) for c in res["citations"]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fall armyworm RAG chatbot with your own sources.")
    ap.add_argument("--kb", default=DEFAULT_KB_DIR, help="knowledge-base folder (default: knowledge_base)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("add-file", help="add PDF/TXT/MD/HTML files"); f.add_argument("paths", nargs="+"); f.add_argument("--title")
    u = sub.add_parser("add-url", help="add a web page or PDF link"); u.add_argument("url"); u.add_argument("--title")
    t = sub.add_parser("add-text", help="add text (from --text or stdin)"); t.add_argument("--title", required=True); t.add_argument("--text")
    sub.add_parser("list", help="list sources")
    r = sub.add_parser("remove", help="remove a source by id"); r.add_argument("id")
    a = sub.add_parser("ask", help="ask a question"); a.add_argument("question")
    sub.add_parser("demo", help="run the notebook's example in a temporary knowledge base")
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bot = MaizeRAGChatbot(tmp)
            bot.load_documents(SAMPLE_SOURCES)
            res = bot.get_chat_response("How do I kill caterpillars on young maize without chemicals?")
            print(json.dumps({k: res[k] for k in ("retrieved_documents", "citations")}, indent=2, ensure_ascii=False))
        return

    bot = MaizeRAGChatbot(args.kb)
    if args.cmd == "add-file":
        for path in args.paths:
            try:
                print("Added:", bot.add_file(path, title=args.title)["title"])
            except Exception as exc:
                print(f"Could not add {path}: {exc}", file=sys.stderr)
    elif args.cmd == "add-url":
        print("Added:", bot.add_url(args.url, args.title)["title"])
    elif args.cmd == "add-text":
        print("Added:", bot.add_text(args.title, args.text or sys.stdin.read())["title"])
    elif args.cmd == "list":
        for s in bot.list_sources():
            print(f"{s['id']}  {s['type']:<4} {s['words']:>6} words  {s['title']}")
        if not bot.list_sources():
            print("No sources yet.")
    elif args.cmd == "remove":
        print("Removed." if bot.remove_source(args.id) else "No source with that id.")
    elif args.cmd == "ask":
        _print_result(bot.answer(args.question))


if _in_streamlit():
    _streamlit_entry()
elif __name__ == "__main__":
    main()
