"""
file_generator.py
==================
Lets Chat mode hand the user a *real* downloadable file — PDF, Word, Excel,
Markdown, plain text, CSV, JSON, HTML, and a handful of other plain-text
formats — instead of only ever printing content into the chat bubble.

How it's triggered
-------------------
The model is instructed (see rules.txt, CHAT MODE section) to only do this
when the user explicitly asked for a downloadable file, by wrapping the
file's content in a pair of sentinel markers that name the file:

    <<<FILE:quarterly-report.pdf>>>
    # Q3 Report
    ...markdown-ish content, which may freely include ``` code fences ```...
    <<<END_FILE>>>

Deliberately NOT a ``` fence: the file's own content is very often markdown
that legitimately contains ``` code blocks (a report with a code sample, a
README, etc.), and a ```file:... wrapper would collide with the first
nested ``` it contained, truncating the file. <<<FILE:...>>> / <<<END_FILE>>>
can't collide with normal prose, markdown, or code.

Two entry points consume that block:

  * `store_file_blocks_and_clean(text, session)` — used for the NON-streaming
    /chat path. Runs once on the model's complete answer: finds every
    <<<FILE:...>>> block, builds+stores the real file, and replaces the raw
    block in the returned text with a short Markdown download link (so the
    existing chat renderer turns it into a clickable link with zero frontend
    changes).

  * `make_stream_watcher(on_token, session, session_id)` — used for the
    STREAMING /chat path. Wraps whatever callback would otherwise receive
    live answer text token-by-token: normal text passes straight through
    unchanged, but once a <<<FILE:...>>> marker opens, the raw content is
    buffered (never forwarded, so the user never sees raw file guts flash by
    mid-generation) until the closing marker, at which point the file is
    built+stored immediately and a Markdown download link is forwarded in
    its place — so the visible stream, and the persisted chat history built
    from it, end up identical to the non-streaming path's output.

Both paths funnel through the same single-file builder, `_build_and_store`,
so there is exactly one place that turns a (filename, raw content) pair into
stored bytes.
"""

import csv as _csv
import io
import re
import uuid
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

MAX_GENERATED_FILE_CHARS = 300_000          # ~300KB of raw text content per file
MAX_GENERATED_FILES_PER_SESSION = 100

# Extensions written straight through as UTF-8 text, no conversion needed.
PASSTHROUGH_EXTS = {
    ".md", ".markdown", ".txt", ".csv", ".json", ".html", ".htm", ".css",
    ".js", ".ts", ".py", ".xml", ".yaml", ".yml", ".sql", ".sh", ".log",
}
# Extensions rendered from markdown-ish (or CSV, for xlsx) text into a real
# binary document.
RENDERED_EXTS = {".pdf", ".docx", ".xlsx"}
SUPPORTED_EXTS = PASSTHROUGH_EXTS | RENDERED_EXTS

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".md": "text/markdown", ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
    ".html": "text/html", ".htm": "text/html",
    ".css": "text/css",
    ".js": "text/javascript", ".ts": "text/plain",
    ".py": "text/x-python",
    ".xml": "application/xml",
    ".yaml": "text/yaml", ".yml": "text/yaml",
    ".sql": "application/sql",
    ".sh": "application/x-sh",
    ".log": "text/plain",
}
DEFAULT_CONTENT_TYPE = "application/octet-stream"

# Matches the opening marker of a file block, e.g. <<<FILE:report.pdf>>>\n
_FILE_FENCE_OPEN_RE = re.compile(
    r"<<<FILE:(?P<name>[A-Za-z0-9_][A-Za-z0-9_.\- ]{0,80})>>>[ \t]*\n?"
)
_CLOSE_MARKER = "<<<END_FILE>>>"
# A full block, opening marker through the matching closing marker, used by
# the one-shot (non-streaming) path.
_FILE_BLOCK_RE = re.compile(
    r"<<<FILE:(?P<name>[A-Za-z0-9_][A-Za-z0-9_.\- ]{0,80})>>>[ \t]*\n?(?P<content>.*?)\n?<<<END_FILE>>>",
    re.DOTALL,
)


def content_type_for(filename: str) -> str:
    ext = _ext(filename)
    return CONTENT_TYPES.get(ext, DEFAULT_CONTENT_TYPE)


def _ext(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot != -1 else ""


def _sanitize_filename(raw: str) -> str:
    """Strip path separators and anything not safe in a Content-Disposition
    header or filesystem-adjacent context. Falls back to a generic name if
    what's left is empty, and forces a supported extension."""
    name = (raw or "").strip().strip("/\\")
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9_.\- ]", "", name).strip(" .") or "file"
    ext = _ext(name)
    if ext not in SUPPORTED_EXTS:
        # Unrecognized/missing extension: keep the base name, default to .txt
        # so the model's content is never silently dropped.
        base = name if not ext else name[: -len(ext)]
        name = (base.strip(" .") or "file") + ".txt"
    return name[:150]


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# ---------------------------------------------------------------------------
# Markdown-ish block parsing, shared by the PDF and DOCX renderers.
# ---------------------------------------------------------------------------

def _parse_markdown_blocks(content: str) -> List[Dict[str, Any]]:
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: List[Dict[str, Any]] = []
    para_buf: List[str] = []
    in_code = False
    code_buf: List[str] = []

    def flush_para() -> None:
        if para_buf:
            blocks.append({"type": "paragraph", "text": " ".join(para_buf).strip()})
            para_buf.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                blocks.append({"type": "code", "text": "\n".join(code_buf)})
                code_buf = []
                in_code = False
            else:
                flush_para()
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue
        if not stripped:
            flush_para()
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_para()
            blocks.append({"type": "heading", "level": len(m.group(1)), "text": m.group(2).strip()})
            continue
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            flush_para()
            blocks.append({"type": "hr"})
            continue
        m = re.match(r"^[-*+]\s+(.*)$", stripped)
        if m:
            flush_para()
            blocks.append({"type": "bullet", "text": m.group(1).strip()})
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            flush_para()
            blocks.append({"type": "numbered", "num": int(m.group(1)), "text": m.group(2).strip()})
            continue
        para_buf.append(stripped)
    flush_para()
    if in_code and code_buf:
        blocks.append({"type": "code", "text": "\n".join(code_buf)})
    return blocks


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_INLINE_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def _inline_to_reportlab_markup(text: str) -> str:
    escaped = _xml_escape(text)
    escaped = _INLINE_CODE_RE.sub(lambda m: f'<font face="Courier">{m.group(1)}</font>', escaped)
    escaped = _INLINE_BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", escaped)
    escaped = _INLINE_ITALIC_RE.sub(lambda m: f"<i>{m.group(1)}</i>", escaped)
    return escaped


_INLINE_TOKEN_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|(?<!\*)\*(?!\*).+?(?<!\*)\*(?!\*))")


def _add_docx_runs(paragraph, text: str) -> None:
    pos = 0
    for m in _INLINE_TOKEN_RE.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos:m.start()])
        token = m.group(0)
        if token.startswith("**"):
            paragraph.add_run(token[2:-2]).bold = True
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
        else:
            paragraph.add_run(token[1:-1]).italic = True
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


# ---------------------------------------------------------------------------
# Renderers: markdown-ish text -> real document bytes
# ---------------------------------------------------------------------------

def _render_pdf(content: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (HRFlowable, Paragraph, Preformatted,
                                         SimpleDocTemplate, Spacer)
    except ImportError as exc:
        raise RuntimeError(
            "PDF generation is unavailable on this server — the 'reportlab' "
            "package is not installed."
        ) from exc

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        topMargin=0.85 * inch, bottomMargin=0.85 * inch,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    )
    styles = getSampleStyleSheet()
    heading_styles = {
        1: ParagraphStyle("GenH1", parent=styles["Heading1"], fontSize=19, spaceBefore=4, spaceAfter=12),
        2: ParagraphStyle("GenH2", parent=styles["Heading2"], fontSize=15, spaceBefore=10, spaceAfter=8),
        3: ParagraphStyle("GenH3", parent=styles["Heading3"], fontSize=12.5, spaceBefore=8, spaceAfter=6),
    }
    body_style = ParagraphStyle("GenBody", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=8)
    bullet_style = ParagraphStyle("GenBullet", parent=body_style, leftIndent=16, spaceAfter=4)
    code_style = ParagraphStyle(
        "GenCode", parent=styles["Code"], fontSize=8.8, leading=11.5,
        backColor="#f2f2f2", borderPadding=6, spaceBefore=4, spaceAfter=10,
    )

    flow = []
    for block in _parse_markdown_blocks(content):
        btype = block["type"]
        if btype == "heading":
            level = min(block["level"], 3)
            flow.append(Paragraph(_inline_to_reportlab_markup(block["text"]), heading_styles.get(level, heading_styles[3])))
        elif btype == "paragraph" and block["text"]:
            flow.append(Paragraph(_inline_to_reportlab_markup(block["text"]), body_style))
        elif btype == "bullet":
            flow.append(Paragraph("\u2022&nbsp;&nbsp;" + _inline_to_reportlab_markup(block["text"]), bullet_style))
        elif btype == "numbered":
            flow.append(Paragraph(f"{block['num']}.&nbsp;&nbsp;" + _inline_to_reportlab_markup(block["text"]), bullet_style))
        elif btype == "hr":
            flow.append(HRFlowable(width="100%", thickness=0.6, spaceBefore=6, spaceAfter=12, color="#cccccc"))
        elif btype == "code" and block["text"]:
            flow.append(Preformatted(block["text"], code_style))
    if not flow:
        flow.append(Spacer(1, 1))
    doc.build(flow)
    return buf.getvalue()


def _render_docx(content: str) -> bytes:
    try:
        import docx as _docx
        from docx.shared import Pt
    except ImportError as exc:
        raise RuntimeError(
            "Word document generation is unavailable on this server — the "
            "'python-docx' package is not installed."
        ) from exc

    document = _docx.Document()
    any_content = False
    for block in _parse_markdown_blocks(content):
        btype = block["type"]
        if btype == "heading":
            document.add_heading(block["text"], level=min(block["level"], 4))
            any_content = True
        elif btype == "paragraph" and block["text"]:
            _add_docx_runs(document.add_paragraph(), block["text"])
            any_content = True
        elif btype == "bullet":
            _add_docx_runs(document.add_paragraph(style="List Bullet"), block["text"])
            any_content = True
        elif btype == "numbered":
            _add_docx_runs(document.add_paragraph(style="List Number"), block["text"])
            any_content = True
        elif btype == "hr":
            document.add_paragraph("_" * 40)
            any_content = True
        elif btype == "code" and block["text"]:
            run = document.add_paragraph().add_run(block["text"])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
            any_content = True
    if not any_content:
        document.add_paragraph("")
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def _render_xlsx(content: str) -> bytes:
    try:
        import openpyxl
        from openpyxl.styles import Font
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError(
            "Excel generation is unavailable on this server — the "
            "'openpyxl' package is not installed."
        ) from exc

    wb = openpyxl.Workbook()
    ws = wb.active
    rows = list(_csv.reader(StringIO(content)))
    widths: Dict[int, int] = {}
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)
            widths[c_idx] = max(widths.get(c_idx, 8), min(len(value) + 2, 60))
    for c_idx, width in widths.items():
        ws.column_dimensions[get_column_letter(c_idx)].width = width
    if rows:
        for c_idx in range(1, len(rows[0]) + 1):
            ws.cell(row=1, column=c_idx).font = Font(bold=True)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def render_bytes(filename: str, raw_content: str) -> bytes:
    ext = _ext(filename)
    if ext == ".pdf":
        return _render_pdf(raw_content)
    if ext == ".docx":
        return _render_docx(raw_content)
    if ext == ".xlsx":
        return _render_xlsx(raw_content)
    return raw_content.encode("utf-8")


# ---------------------------------------------------------------------------
# Session-scoped storage (mirrors rag_engine.py's upload-file store, kept
# separate since these are model-authored deliverables, not user uploads).
# ---------------------------------------------------------------------------

def _store(session: dict) -> Dict[str, dict]:
    return session.setdefault("generated_files", {})


def list_generated_files(session: dict) -> List[dict]:
    files = session.get("generated_files") or {}
    return sorted(
        ({k: v for k, v in rec.items() if not k.startswith("_")} for rec in files.values()),
        key=lambda f: f.get("created_at", ""),
    )


def get_generated_file(session: dict, file_id: str) -> Optional[Tuple[bytes, str, str]]:
    record = (session.get("generated_files") or {}).get(file_id)
    if not record:
        return None
    data = record.get("_data")
    if not isinstance(data, (bytes, bytearray)):
        return None
    return bytes(data), record.get("content_type") or DEFAULT_CONTENT_TYPE, record.get("filename") or "download"


def _build_and_store(session: dict, session_id: str, filename: str, raw_content: str) -> Dict[str, Any]:
    """The single place a (filename, content) pair becomes a stored,
    downloadable file. Returns public metadata plus a ready-to-render
    Markdown link for the chat bubble; on failure returns an `error` field
    and a Markdown line explaining what went wrong instead of raising, so a
    render problem degrades to a visible note rather than breaking the turn."""
    clean_name = _sanitize_filename(filename)
    files = _store(session)
    if len(files) >= MAX_GENERATED_FILES_PER_SESSION:
        return {"error": True, "markdown": f"\n\n_Could not create **{clean_name}** — this session has reached its file limit._\n"}
    trimmed_content = raw_content[:MAX_GENERATED_FILE_CHARS]
    try:
        data = render_bytes(clean_name, trimmed_content)
    except Exception as exc:
        return {"error": True, "markdown": f"\n\n_Could not create **{clean_name}**: {str(exc)[:200]}_\n"}

    file_id = uuid.uuid4().hex[:12]
    record = {
        "id": file_id,
        "filename": clean_name,
        "size": len(data),
        "content_type": content_type_for(clean_name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "_data": data,
    }
    files[file_id] = record
    # Keep the cosmetic filename segment URL-safe. The download endpoint uses
    # file_id as the source of truth, but an unescaped space would break the
    # Markdown link and the frontend's generated-file-card parser.
    download_url = f"/generated-files/{session_id}/{file_id}/{quote(clean_name)}"
    markdown = f"\n\n\U0001F4CE **[{clean_name}]({download_url})** ({human_size(len(data))}) — ready to download.\n"
    return {"error": False, "id": file_id, "filename": clean_name, "size": len(data), "download_url": download_url, "markdown": markdown}


# ---------------------------------------------------------------------------
# Non-streaming entry point: one-shot find, build, and replace on a
# complete response string.
# ---------------------------------------------------------------------------

def store_file_blocks_and_clean(text: str, session: dict, session_id: str) -> str:
    if "<<<FILE:" not in (text or ""):
        return text

    def _replace(match: "re.Match") -> str:
        filename = match.group("name").strip()
        content = match.group("content")
        result = _build_and_store(session, session_id, filename, content)
        return result["markdown"]

    return _FILE_BLOCK_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Streaming entry point: wraps a live token callback so raw file content
# never reaches the visible stream, building+storing the real file the
# instant its closing fence is seen.
# ---------------------------------------------------------------------------

def make_stream_watcher(on_token, session: dict, session_id: str):
    """Returns an async `watcher(text)` suitable as `invoke_model`'s
    `on_answer_piece`. Ordinary text is forwarded to `on_token` unchanged
    (after a small holdback so a marker split across two chunks is never
    missed); text inside a <<<FILE:...>>> block is buffered and never
    forwarded — a Markdown download link takes its place once the block
    closes."""
    state = {"buffer": "", "mode": "text", "filename": None, "capture": ""}
    HOLD = 90  # long enough to never lose a partial "<<<FILE:name.ext>>>" tag
    CLOSE_HOLD = len(_CLOSE_MARKER) - 1

    async def _emit(text: str) -> None:
        if text:
            await on_token(text)

    async def watcher(text: str) -> None:
        if not text:
            return
        state["buffer"] += text
        while True:
            if state["mode"] == "text":
                m = _FILE_FENCE_OPEN_RE.search(state["buffer"])
                if m:
                    pre = state["buffer"][:m.start()]
                    if pre:
                        await _emit(pre)
                    state["filename"] = m.group("name").strip()
                    state["buffer"] = state["buffer"][m.end():]
                    state["mode"] = "capture"
                    state["capture"] = ""
                    continue  # re-run loop in capture mode on the remainder
                hold_back = min(len(state["buffer"]), HOLD)
                send_len = len(state["buffer"]) - hold_back
                if send_len > 0:
                    await _emit(state["buffer"][:send_len])
                    state["buffer"] = state["buffer"][send_len:]
                return
            # mode == "capture"
            close_idx = state["buffer"].find(_CLOSE_MARKER)
            if close_idx == -1:
                hold_back = min(len(state["buffer"]), CLOSE_HOLD)
                send_len = len(state["buffer"]) - hold_back
                if send_len > 0:
                    state["capture"] += state["buffer"][:send_len]
                    state["buffer"] = state["buffer"][send_len:]
                return
            state["capture"] += state["buffer"][:close_idx]
            remainder = state["buffer"][close_idx + len(_CLOSE_MARKER):]
            state["buffer"] = ""
            content = state["capture"].lstrip("\n")
            if content.endswith("\n"):
                content = content[:-1]
            result = _build_and_store(session, session_id, state["filename"], content)
            await _emit(result["markdown"])
            state["mode"] = "text"
            state["capture"] = ""
            state["filename"] = None
            if remainder:
                state["buffer"] = remainder
                continue
            return

    async def finalize() -> None:
        """Call once after generation ends to flush any text held back for
        fence-boundary safety, or to salvage an unterminated file block
        (model forgot the closing fence) rather than losing the content."""
        if state["mode"] == "text":
            if state["buffer"]:
                await _emit(state["buffer"])
                state["buffer"] = ""
        else:
            content = (state["capture"] + state["buffer"]).lstrip("\n")
            state["buffer"] = ""
            state["capture"] = ""
            if content or state["filename"]:
                result = _build_and_store(session, session_id, state["filename"] or "file.txt", content)
                await _emit(result["markdown"])
            state["mode"] = "text"

    return watcher, finalize
