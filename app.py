"""Comprehension-gap annotation tool.

Team members open the app, enter their name, step through the threads of an
input JSONL file and record for each one:
  1. whether a comprehension gap exists (True / False)
  2. the gap category, picked from a shared hierarchical taxonomy
     (main category > sub-category > ...); anyone can add new nodes
  3. optional notes

Every saved review is written to OUTPUT_PATH (JSONL): one line per
(thread_id, annotator), holding the original record plus a `human_review` block.
Saving the same thread again overwrites that annotator's earlier line.
"""

import hashlib
import html
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from filelock import FileLock

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_DIR / "input.jsonl"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", DATA_DIR / "annotations.jsonl"))
TAXONOMY_PATH = Path(os.environ.get("TAXONOMY_PATH", DATA_DIR / "taxonomy.json"))

# Base main categories. The team can add sub-categories (any depth) and new
# main categories from the app; those are stored in TAXONOMY_PATH.
BASE_CATEGORIES = [
    ("CODE_BEHAVIOR",
     "What did the contributor misunderstand about how the submitted code executes "
     "or what result it produces?",
     "I thought it was doing this, but I was wrong. Misunderstandings about edge cases."),
    ("SCOPE_AND_CONTEXT",
     "What did the contributor misunderstand about what was in the patch, what was already "
     "in the system, or which parts of the system the patch affected?",
     "Didn't realize ripple effects in the system; unaware of an existing feature; "
     "unaware of a change introduced by the patch."),
    ("INTENT_AND_RATIONALE",
     "What did the contributor misunderstand about the goal of the change or why the "
     "implementation was chosen?",
     "I had to add this statement to make the test pass, but don't know why. "
     "Everything related to requirements."),
    ("VALIDATION",
     "What did the contributor misunderstand about what a test or check establishes?",
     "I thought the test checked this, indicated that, etc."),
]
SEP = "/"  # category ids are paths: "VALIDATION/test_scope_misread"

st.set_page_config(page_title="Gap Annotator", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------- file helpers

def _lock(path: Path) -> FileLock:
    return FileLock(str(path) + ".lock", timeout=10)


def read_jsonl(path_or_bytes) -> list:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        lines = path_or_bytes.decode("utf-8").splitlines()
    else:
        if not Path(path_or_bytes).exists():
            return []
        lines = Path(path_or_bytes).read_text(encoding="utf-8").splitlines()
    records, bad = [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        st.warning(f"Skipped {bad} malformed line(s).")
    return records


def write_jsonl_atomic(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------------ lazy input index
# The input file can be huge (GBs), so it is never loaded into memory. We scan it
# once and keep, per line, its byte offset plus the few fields needed for
# filtering / navigation / the timeline. A full record is read from disk only
# when it is displayed. The index is cached in memory (shared by all sessions)
# and persisted next to the input file, so restarts don't rescan.

INDEX_VERSION = 2
INDEX_FIELDS = ("offset", "thread_id", "repo", "pr_number", "positive", "date", "ai_prov")


def _index_meta(rec: dict) -> tuple:
    d = thread_date(rec)
    return (str(rec.get("thread_id")), rec.get("repo", "?"), rec.get("pr_number", "?"),
            model_positive(rec), d.isoformat() if d else None, ai_provenance_status(rec))


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".idx.json")


@st.cache_resource(show_spinner=False, max_entries=4)
def build_index(path_str: str, size: int, mtime: float) -> dict:
    """Column-oriented index of the JSONL file at `path_str` (size/mtime bust the cache)."""
    path = Path(path_str)
    side = _sidecar(path)
    try:
        cached = json.loads(side.read_text(encoding="utf-8"))
        if (cached.get("version"), cached.get("size"), cached.get("mtime")) == (INDEX_VERSION, size, mtime):
            return cached
    except (OSError, ValueError):
        pass

    cols = {f: [] for f in INDEX_FIELDS}
    bad, offset = 0, 0
    bar = st.progress(0.0, text=f"Indexing {path.name} (one-time)…")
    step = max(size // 100, 1)
    next_tick = step
    with open(path, "rb") as f:
        for line in f:
            start, offset = offset, offset + len(line)
            if line.strip():
                try:
                    meta = (start,) + _index_meta(json.loads(line))
                except (ValueError, AttributeError):
                    bad += 1
                else:
                    for k, v in zip(INDEX_FIELDS, meta):
                        cols[k].append(v)
            if offset >= next_tick:
                bar.progress(min(offset / size, 1.0), text=f"Indexing {path.name} (one-time)… "
                             f"{len(cols['offset']):,} threads")
                next_tick += step
    bar.empty()

    index = {"version": INDEX_VERSION, "size": size, "mtime": mtime, "bad": bad, **cols}
    try:
        fd, tmp = tempfile.mkstemp(dir=side.parent, prefix=".tmp_idx_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(index, fh)
        os.replace(tmp, side)
    except OSError:
        pass  # read-only dir: keep the in-memory index only
    return index


class LazyRecords:
    """List-like view over a JSONL file: len(), meta lookups, and on-demand records."""

    def __init__(self, path: Path):
        st_ = path.stat()
        self.path = path
        self.idx = build_index(str(path), st_.st_size, st_.st_mtime)
        if self.idx.get("bad"):
            st.sidebar.warning(f"Skipped {self.idx['bad']} malformed line(s) in {path.name}.")

    def __len__(self) -> int:
        return len(self.idx["offset"])

    def meta(self, field: str, i: int):
        return self.idx[field][i]

    def __getitem__(self, i: int) -> dict:
        return _read_record(str(self.path), self.idx["offset"][i], self.idx["size"])


@st.cache_resource(show_spinner=False, max_entries=4)
def _row_lookup(path_str: str, size: int, mtime: float) -> dict:
    idx = build_index(path_str, size, mtime)
    return {tid: i for i, tid in enumerate(idx["thread_id"])}


def thread_row_lookup(records: "LazyRecords") -> dict:
    """thread_id -> row index (built once per input file, shared by all sessions)."""
    return _row_lookup(str(records.path), records.idx["size"], records.idx["mtime"])


@st.cache_data(show_spinner=False, max_entries=256)
def _read_record(path_str: str, offset: int, _size: int) -> dict:
    with open(path_str, "rb") as f:
        f.seek(offset)
        return json.loads(f.readline())


def upload_to_file(data: bytes) -> Path:
    """Store an uploaded JSONL under a content-hash name so it can be indexed like a server file."""
    d = Path(tempfile.gettempdir()) / "gapannotator_uploads"
    d.mkdir(exist_ok=True)
    p = d / f"{hashlib.sha1(data).hexdigest()}.jsonl"
    if not p.exists():
        p.write_bytes(data)
    return p


def load_annotations() -> list:
    with _lock(OUTPUT_PATH):
        return read_jsonl(OUTPUT_PATH)


def upsert_annotation(record: dict) -> None:
    """Replace this annotator's line for this thread, or append a new one."""
    key = (str(record["thread_id"]), record["human_review"]["annotator"])
    with _lock(OUTPUT_PATH):
        rows = read_jsonl(OUTPUT_PATH)
        rows = [
            r for r in rows
            if (str(r.get("thread_id")), r.get("human_review", {}).get("annotator")) != key
        ]
        rows.append(record)
        write_jsonl_atomic(OUTPUT_PATH, rows)


# ------------------------------------------------------------ category taxonomy
# Stored as a flat list of nodes; each node's id is its full path, so the
# parent is everything before the last SEP.

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_taxonomy(nodes: list) -> None:
    TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=TAXONOMY_PATH.parent, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, TAXONOMY_PATH)


def load_taxonomy() -> list:
    with _lock(TAXONOMY_PATH):
        if TAXONOMY_PATH.exists():
            return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["nodes"]
        nodes = [{"id": cid, "description": d, "examples": ex, "created_by": "base",
                  "created_at": _now()} for cid, d, ex in BASE_CATEGORIES]
        _write_taxonomy(nodes)
        return nodes


def normalize_name(name: str, top_level: bool) -> str:
    s = "_".join(name.replace(SEP, " ").strip().split())
    return s.upper() if top_level else s.lower()


# ----------------------------------------------------------- thread rendering

ROLE_LABELS = {"pr_author": "Author", "reviewer": "Reviewer"}

_DIFF_STYLES = {  # line kind -> (row background, gutter background)
    "+": ("rgba(46,160,67,0.15)", "rgba(46,160,67,0.30)"),
    "-": ("rgba(248,81,73,0.15)", "rgba(248,81,73,0.30)"),
    " ": ("transparent", "transparent"),
    "@": ("rgba(56,139,253,0.10)", "rgba(56,139,253,0.18)"),
}


def render_diff_hunk(hunk: str) -> str:
    """Render a unified-diff hunk as a GitHub-like table with old/new line numbers."""
    m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", hunk)
    old_no, new_no = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    rows = []
    for line in hunk.splitlines():
        kind = line[:1] if line[:1] in "+- @" else " "
        text = line if kind == "@" else line[1:]
        if kind == "@":
            old_txt = new_txt = "…"
        elif kind == "-":
            old_txt, new_txt = old_no, ""
            old_no = old_no + 1 if old_no is not None else None
        elif kind == "+":
            old_txt, new_txt = "", new_no
            new_no = new_no + 1 if new_no is not None else None
        else:
            old_txt, new_txt = old_no, new_no
            old_no = old_no + 1 if old_no is not None else None
            new_no = new_no + 1 if new_no is not None else None
        bg, gutter = _DIFF_STYLES[kind]
        num = ("padding:0 8px;text-align:right;color:#888;user-select:none;"
               f"background:{gutter};min-width:3em")
        sign = "" if kind == "@" else kind
        color = "color:#4c8bf5;" if kind == "@" else ""
        rows.append(
            f'<tr style="background:{bg}">'
            f'<td style="{num}">{"" if old_txt is None else old_txt}</td>'
            f'<td style="{num}">{"" if new_txt is None else new_txt}</td>'
            f'<td style="padding:0 6px;user-select:none;width:1em">{sign}</td>'
            f'<td style="white-space:pre;{color}">{html.escape(text)}</td></tr>')
    return ('<div style="overflow-x:auto;border:1px solid rgba(128,128,128,0.3);border-radius:6px">'
            '<table style="border-collapse:collapse;width:100%;font-family:ui-monospace,'
            'SFMono-Regular,Menlo,monospace;font-size:12.5px;line-height:1.6;'
            'font-variant-ligatures:none;font-feature-settings:\'liga\' 0,\'calt\' 0">'
            + "".join(rows) + "</table></div>")


def literal_md(text: str) -> str:
    """Streamlit's markdown turns >=, <=, !=, -> etc. into ≥ ≤ ≠ →, which changes
    what reviewers wrote. Break them up with an invisible word joiner (U+2060)
    outside code spans/blocks and URLs; backslash escapes don't stop the replacement."""
    parts = re.split(r"(```[\s\S]*?```|`[^`\n]*`|https?://\S+)", (text or "").replace("\r\n", "\n"))
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"([<>!=-])(?=[=<>-])", "\\1\u2060", parts[i])
    return "".join(parts)


def thread_comments(rec: dict, ann: dict) -> list:
    """Every comment of the thread (from `thread_conversation`), plus any comment
    only mentioned in the annotation, de-duplicated by comment id and sorted by
    time. Comments the model used as a reviewer question or as gap evidence are
    flagged so they can be tagged."""
    by_id = {}

    def get(cid, **fields):
        c = by_id.setdefault(str(cid), {"id": str(cid), "is_question": False, "evidence_for": [],
                                        "question_meta": [], "body": "", "author": "?",
                                        "role": "", "created_at": "", "url": ""})
        c.update({k: v for k, v in fields.items() if v})
        return c

    for m in rec.get("thread_conversation") or []:
        get(m.get("comment_id"), body=m.get("body"), author=m.get("author"),
            role=m.get("role"), created_at=m.get("created_at"), url=m.get("html_url"))
    for q in ann.get("reviewer_questions", []):
        c = get(q.get("question_comment_id"), body=q.get("question_body"),
                author=q.get("question_author"), role=q.get("question_role"),
                created_at=q.get("question_created_at"), url=q.get("question_html_url"))
        c["is_question"] = True
        c["question_meta"].append(q)
        for r in q.get("contributor_responses", []):
            get(r.get("comment_id"), body=r.get("body"), author=r.get("author"),
                role=r.get("role"), created_at=r.get("created_at"), url=r.get("html_url"))
    for g in ann.get("gaps", []):
        for ev in g.get("evidence", []):
            c = get(ev.get("comment_id"), body=ev.get("body"), author=ev.get("author"),
                    role=ev.get("role"), created_at=ev.get("created_at"), url=ev.get("html_url"))
            if g.get("gap_id") not in c["evidence_for"]:
                c["evidence_for"].append(g.get("gap_id"))
    return sorted(by_id.values(), key=lambda c: c["created_at"] or "")


def add_category(parent: str, name: str, description: str, author: str) -> str:
    cid = (parent + SEP if parent else "") + normalize_name(name, not parent)
    with _lock(TAXONOMY_PATH):
        nodes = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["nodes"]
        if not any(n["id"] == cid for n in nodes):
            nodes.append({"id": cid, "description": description.strip(), "examples": "",
                          "created_by": author, "created_at": _now()})
            _write_taxonomy(nodes)
    return cid


def delete_category(cid: str) -> None:
    with _lock(TAXONOMY_PATH):
        nodes = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["nodes"]
        _write_taxonomy([n for n in nodes if n["id"] != cid])


def sort_tree(nodes: list) -> list:
    """Depth-first order: each parent followed by its children."""
    return sorted(nodes, key=lambda n: n["id"].split(SEP))


def pretty(cid: str) -> str:
    return cid.replace(SEP, "  ›  ")


def usage_counts(annotations: list) -> dict:
    """Reviews per category, counting a review once for each ancestor too."""
    counts = {}
    for a in annotations:
        hit = set()
        for path in a.get("human_review", {}).get("gap_categories", []):
            parts = path.split(SEP)
            hit.update(SEP.join(parts[:i]) for i in range(1, len(parts) + 1))
        for c in hit:
            counts[c] = counts.get(c, 0) + 1
    return counts


# ------------------------------------------------------------- timeline page

def model_positive(r: dict) -> bool:
    flag = (r.get("annotation") or {}).get("has_comprehension_debt")
    return flag is True or str(flag).strip().lower() == "true"


def ai_provenance_status(rec: dict) -> str:
    """The model's AI-provenance status for a thread, as a display string."""
    prov = (rec.get("annotation") or {}).get("ai_provenance")
    status = prov.get("status") if isinstance(prov, dict) else prov
    return str(status) if status not in (None, "") else "(missing)"


def thread_date(rec: dict):
    """When the thread started: its earliest comment timestamp (UTC), or None."""
    stamps = [m.get("created_at") for m in rec.get("thread_conversation") or []
              if m.get("created_at")]
    if not stamps:
        return None
    try:
        return datetime.fromisoformat(min(stamps).replace("Z", "+00:00"))
    except ValueError:
        return None


def human_gap_votes(annotations: list) -> dict:
    """thread_id -> list of gap_exists votes from human reviewers."""
    votes = {}
    for a in annotations:
        hr = a.get("human_review") or {}
        if "gap_exists" in hr:
            votes.setdefault(str(a.get("thread_id")), []).append(bool(hr["gap_exists"]))
    return votes


def timeline_page(records: "LazyRecords") -> None:
    import altair as alt
    import pandas as pd

    st.header("📈 Comprehension gaps over time")
    st.caption("Each thread is dated by its first comment. Shows only positive threads "
               "(with a comprehension gap) per period.")

    c1, c2, c3 = st.columns([2, 2, 3])
    label_src = c1.radio("Gap label from", ["Model", "Human reviews"], horizontal=True,
                         help="Human reviews: majority vote of everyone who reviewed the "
                              "thread (ties count as gap). Unreviewed threads are skipped.")
    grain = c2.radio("Group by", ["Month", "Quarter", "Year"], horizontal=True)
    idx = records.idx
    repos = sorted(set(idx["repo"]))
    picked = set(c3.multiselect("Repositories", repos, placeholder="All repositories"))

    votes = human_gap_votes(load_annotations()) if label_src == "Human reviews" else {}
    rows, undated = [], 0
    for tid, repo, positive, date in zip(idx["thread_id"], idx["repo"], idx["positive"], idx["date"]):
        if picked and repo not in picked:
            continue
        if label_src == "Model":
            gap = positive
        else:
            v = votes.get(tid)
            if not v:
                continue
            gap = sum(v) * 2 >= len(v)
        if not gap:
            continue
        if date is None:
            undated += 1
            continue
        rows.append({"date": datetime.fromisoformat(date).replace(tzinfo=None), "gap": gap,
                     "repo": repo, "thread_id": tid})

    if not rows:
        st.info("No dated positive threads match this selection."
                + (" No human reviews saved yet." if label_src == "Human reviews" and not votes else ""))
        return

    df = pd.DataFrame(rows)
    freq = {"Month": "M", "Quarter": "Q", "Year": "Y"}[grain]
    df["period"] = df["date"].dt.to_period(freq).dt.start_time
    agg = df.groupby("period").size().rename("gapped").reset_index()
    # Fill empty periods with zeros so gaps in time are visible, not skipped.
    full = pd.period_range(df["date"].min(), df["date"].max(), freq=freq).start_time
    agg = agg.set_index("period").reindex(full, fill_value=0).rename_axis("period").reset_index()
    agg["gapped"] = agg["gapped"].astype(int)
    fmt = {"Month": "%b %Y", "Quarter": "%Y-Q", "Year": "%Y"}[grain]
    if grain == "Quarter":
        agg["label"] = agg["period"].dt.year.astype(str) + "-Q" + agg["period"].dt.quarter.astype(str)
    else:
        agg["label"] = agg["period"].dt.strftime(fmt)

    n_gap = int(agg["gapped"].sum())
    m = st.columns(3)
    m[0].metric("🟢 Positive threads", f"{n_gap:,}")
    m[1].metric(f"Busiest {grain.lower()}", agg.loc[agg["gapped"].idxmax(), "label"])
    m[2].metric(f"Avg per {grain.lower()}", f"{agg['gapped'].mean():.1f}")
    if undated:
        st.caption(f"{undated} positive thread(s) without comment timestamps are left out.")

    # ---- distribution of positives over time
    x = alt.X("period:T", title=grain, axis=alt.Axis(format=fmt if grain != "Quarter" else "%Y-%m"))
    tip = [alt.Tooltip("label:N", title=grain), alt.Tooltip("gapped:Q", title="Positive threads")]
    bars = alt.Chart(agg).mark_bar(color="#16a34a", opacity=0.85).encode(
        x=x, y=alt.Y("gapped:Q", title="Positive threads"), tooltip=tip)
    text = alt.Chart(agg[agg["gapped"] > 0]).mark_text(dy=-6, fontSize=10, color="#64748b").encode(
        x=x, y="gapped:Q", text="gapped:Q")
    st.subheader("Positive threads per period")
    st.altair_chart((bars + text).properties(height=380), use_container_width=True)

    # ---- table with change vs. previous period
    st.subheader("Per-period numbers")
    table = agg[["label", "gapped"]].copy()
    table["Δ vs. previous"] = table["gapped"].diff()
    table["Share of all positives"] = (table["gapped"] / n_gap).round(3)
    table = table.rename(columns={"label": grain, "gapped": "🟢 Positive threads"})
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"Share of all positives": st.column_config.NumberColumn(format="percent")})


# ------------------------------------------------------------------- sidebar

st.sidebar.title("🔎 Gap Annotator")
annotator = st.sidebar.text_input("Your name", key="annotator").strip()

source = st.sidebar.radio("Input data", ["Server file", "Upload JSONL"], horizontal=True)
if source == "Upload JSONL":
    up = st.sidebar.file_uploader("JSONL file", type=["jsonl", "json", "txt"])
    records = LazyRecords(upload_to_file(up.getvalue())) if up else []
    if up and st.sidebar.button("Use as server input file for the team"):
        INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        INPUT_PATH.write_bytes(up.getvalue())
        st.sidebar.success(f"Saved to {INPUT_PATH}")
else:
    records = LazyRecords(INPUT_PATH) if INPUT_PATH.exists() else []
    st.sidebar.caption(f"`{INPUT_PATH}`")

if not records:
    st.info("No input records yet. Upload a JSONL file in the sidebar, or place one at "
            f"`{INPUT_PATH}` on the server.")
    st.stop()

page = st.sidebar.radio("Page", ["📝 Annotate", "📈 Timeline"], horizontal=True)
if page == "📈 Timeline":
    timeline_page(records)
    st.stop()

if not annotator:
    st.info("👈 Enter your name in the sidebar to start annotating.")
    st.stop()

annotations = load_annotations()
mine = {
    str(a["thread_id"]): a["human_review"]
    for a in annotations
    if a.get("human_review", {}).get("annotator") == annotator
}
reviewers_by_thread = {}
for a in annotations:
    reviewers_by_thread.setdefault(str(a.get("thread_id")), []).append(
        a.get("human_review", {}).get("annotator"))

POS, TIDS = records.idx["positive"], records.idx["thread_id"]
n_pos = sum(POS)
label_options = {
    f"Positives only ({n_pos:,})": lambda i: POS[i],
    f"Negatives only ({len(records) - n_pos:,})": lambda i: not POS[i],
    f"Mixed ({len(records):,})": lambda i: True,
}
label_choice = st.sidebar.radio(
    "Model label to review", list(label_options), index=0,
    help="Positive = the model set has_comprehension_debt to true.")
keep = label_options[label_choice]

PROV = records.idx["ai_prov"]
prov_counts = {}
for i in range(len(records)):
    if keep(i):
        prov_counts[PROV[i]] = prov_counts.get(PROV[i], 0) + 1
prov_pick = set(st.sidebar.multiselect(
    "AI provenance", sorted(prov_counts), placeholder="All statuses",
    format_func=lambda v: f"{v} ({prov_counts[v]:,})",
    help="The model's annotation.ai_provenance.status. Counts are within the model-label "
         "choice above. Leave empty to include all."))
pool = [i for i in range(len(records)) if keep(i) and (not prov_pick or PROV[i] in prov_pick)]

done = sum(1 for i in pool if TIDS[i] in mine)
st.sidebar.progress(done / len(pool) if pool else 0.0,
                    text=f"You: {done:,} / {len(pool):,} reviewed")
st.sidebar.caption(f"Team total: {len(annotations)} reviews saved")

show = st.sidebar.selectbox("Show", ["All", "Not reviewed by me", "Reviewed by me"])
if show == "Not reviewed by me":
    visible = [i for i in pool if TIDS[i] not in mine]
elif show == "Reviewed by me":
    visible = [i for i in pool if TIDS[i] in mine]
else:
    visible = pool

if not visible:
    st.success("Nothing to show with this filter. 🎉")
    st.stop()

if "pos" not in st.session_state:
    st.session_state.pos = 0
st.session_state.pos = min(st.session_state.pos, len(visible) - 1)


def label_for(i: int) -> str:
    mark = "✅" if TIDS[i] in mine else "⬜"
    return (f"{mark} #{i + 1} · {records.meta('repo', i)} PR {records.meta('pr_number', i)} "
            f"· {TIDS[i]}")


# --- Find a thread: by thread ID, GitHub URL (…discussion_r<ID>) or row number (#N).
# Handled before "Jump to" is drawn so it can move st.session_state.pos.
ROW_OF = thread_row_lookup(records)


def resolve_query(q: str):
    """Return (row index, None) or (None, error message)."""
    q = q.strip()
    m = re.search(r"discussion_r(\d+)", q)
    if m:
        q = m.group(1)
    if q in ROW_OF:
        return ROW_OF[q], None
    num = q.lstrip("#").strip()
    if num.isdigit() and 1 <= int(num) <= len(records):
        return int(num) - 1, None
    return None, f"No thread with ID or row number `{q}`."


if "forced_row" not in st.session_state:
    st.session_state.forced_row = None
if st.session_state.pop("find_pending", False) and st.session_state.find.strip():
    row, err = resolve_query(st.session_state.find)
    if err:
        st.sidebar.error(err)
    else:
        visible_pos = {r: p for p, r in enumerate(visible)}
        if row in visible_pos:  # inside the current filter: normal navigation from there
            st.session_state.pos = visible_pos[row]
            st.session_state.forced_row = None
        else:  # filtered out: still open it, Prev/Next go back to the list
            st.session_state.forced_row = row

st.sidebar.text_input("🔍 Find thread", key="find", placeholder="Thread ID, GitHub link, or #row",
                      on_change=lambda: st.session_state.update(find_pending=True),
                      help="Exact thread ID (e.g. 681825717), its GitHub discussion link, "
                           "or the row number shown as #N in Jump to. Press Enter.")

# A selectbox with 100k+ options would freeze the browser, so "Jump to" only
# lists a window around the current position; "Find thread" reaches anything else.
JUMP_WINDOW = 100
lo = max(0, st.session_state.pos - JUMP_WINDOW)
hi = min(len(visible), st.session_state.pos + JUMP_WINDOW + 1)
st.sidebar.selectbox(
    "Jump to", range(lo, hi), format_func=lambda p: label_for(visible[p]), key="pos",
    on_change=lambda: st.session_state.update(forced_row=None))

with st.sidebar.expander("Download"):
    if OUTPUT_PATH.exists():
        st.download_button("annotations.jsonl", OUTPUT_PATH.read_bytes(),
                           file_name="annotations.jsonl", mime="application/jsonl")
    else:
        st.caption("No annotations saved yet.")


def go(delta: int) -> None:
    st.session_state.forced_row = None
    st.session_state.pos = max(0, min(len(visible) - 1, st.session_state.pos + delta))


# ---------------------------------------------------------------------- main

forced = st.session_state.forced_row
idx = forced if forced is not None else visible[st.session_state.pos]
rec = records[idx]
if forced is not None:
    st.info(f"Showing searched thread #{idx + 1}. It is not in your current sidebar filter "
            f"(model label / Show), so Prev / Next return to the filtered list.")
tid = str(rec.get("thread_id"))
ann = rec.get("annotation", {}) or {}

nav1, nav2, nav3 = st.columns([1, 6, 1])
nav1.button("◀ Prev", on_click=go, args=(-1,), use_container_width=True,
            disabled=st.session_state.pos == 0)
nav3.button("Next ▶", on_click=go, args=(1,), use_container_width=True,
            disabled=st.session_state.pos >= len(visible) - 1)
nav2.markdown(
    f"### {rec.get('repo', '')} · PR #{rec.get('pr_number', '')} "
    f"<span style='font-size:0.6em;color:gray'>(#{idx + 1}"
    + ("" if forced is not None else f" · {st.session_state.pos + 1} of {len(visible):,} in list")
    + ")</span>",
    unsafe_allow_html=True)

meta = st.columns(4)
meta[0].markdown(f"**Thread** [{tid}]({rec['html_url']})" if rec.get("html_url") else f"**Thread** `{tid}`")
meta[1].markdown(f"**Author** `{rec.get('pr_author', '')}`")
meta[2].markdown(f"**File** `{rec.get('path', '')}`")
if rec.get("html_url"):
    meta[3].markdown(f"[Open on GitHub ↗]({rec['html_url']})")

others = [a for a in reviewers_by_thread.get(tid, []) if a != annotator]
if others:
    st.caption("Also reviewed by: " + ", ".join(sorted(set(others))))

left, right = st.columns([3, 2], gap="large")

with left:
    t_read, t_tree, t_raw = st.tabs(["📖 Readable", "🌳 JSON tree", "{ } Raw"])

    with t_read:
        debt_types = ", ".join(f"`{t}`" for t in ann.get("debt_types") or []) or "—"
        st.markdown(
            f"**Model says** — comprehension debt: `{ann.get('has_comprehension_debt')}` · "
            f"debt types: {debt_types} · "
            f"evidence tier: `{ann.get('evidence_tier')}` · "
            f"AI provenance: `{(ann.get('ai_provenance') or {}).get('status')}`")

        # GitHub-style review thread: file header, diff hunk, then the comments.
        with st.container(border=True):
            st.markdown(f"**`{rec.get('path', '')}`**")
            if rec.get("diff_hunk"):
                st.html(render_diff_hunk(rec["diff_hunk"]))
            comments = thread_comments(rec, ann)
            st.caption(f"{len(comments)} comment(s) in this thread")
            for c in comments:
                with st.container(border=True):
                    badges = [f":blue-badge[{ROLE_LABELS.get(c['role'], c['role'] or '?')}]"]
                    if c["is_question"]:
                        badges.append(":violet-badge[❓ question]")
                    badges += [f":orange-badge[🧾 evidence {g}]" for g in c["evidence_for"]]
                    link = f" · [{c['id']}]({c['url']})" if c.get("url") else ""
                    st.markdown(f"**{c['author']}** on {c['created_at'][:10]}{link} &nbsp; "
                                + " ".join(badges))
                    st.markdown(literal_md(c["body"]))
                    for q in c["question_meta"]:
                        st.caption(f"response: `{q.get('response_status')}` · "
                                   f"correctness: `{q.get('answer_correctness')}` · "
                                   f"linked gaps: {', '.join(q.get('linked_gap_ids') or []) or '—'}"
                                   + (f" · note: {q['note']}" if q.get("note") else ""))

        # Tint the model-gap cards green (Streamlit adds class st-key-<key> to keyed containers).
        st.html("""<style>
            div[class*="st-key-gapcard_"] {
                background: rgba(46,160,67,0.10);
                border: 1px solid rgba(46,160,67,0.55) !important;
                border-left: 5px solid rgb(46,160,67) !important;
                border-radius: 8px;
            }
            .gap-facts {
                display: grid; grid-template-columns: max-content 1fr; gap: 6px 12px;
                margin-top: 4px; padding: 10px 12px; border-radius: 6px;
                background: rgba(46,160,67,0.14); font-size: 0.95rem;
            }
            .gap-facts .k { font-weight: 600; opacity: 0.8; }
            .gap-facts code { font-size: 0.9rem; padding: 1px 6px; border-radius: 4px;
                              background: rgba(128,128,128,0.18); }
        </style>""")
        for n, g in enumerate(ann.get("gaps", [])):
            with st.container(border=True, key=f"gapcard_{tid}_{n}"):
                types = " ".join(f":orange-badge[{t}]" for t in g.get("types") or [])
                if g.get("new_type_name"):
                    types += f" :green-badge[new: {g['new_type_name']}]"
                st.markdown(f"**🕳️ Model gap {g.get('gap_id')}** {types} — subject "
                            f"`{g.get('subject_actor_id')}` · basis `{g.get('evidence_basis')}`")
                st.markdown(literal_md(g.get("gap_description", "")))
                ev_refs = ", ".join(
                    f"<b>{html.escape(str(ev.get('author')))}</b> "
                    f"<code>{html.escape(str(ev.get('comment_id')))}</code>"
                    for ev in g.get("evidence", [])) or "—"
                st.html(
                    '<div class="gap-facts">'
                    '<span class="k">Understanding at end</span>'
                    f"<span><code>{html.escape(str(g.get('understanding_at_end')))}</code></span>"
                    '<span class="k">🧾 Evidence</span>'
                    f"<span>{ev_refs} <i style='opacity:0.7'>(tagged in the thread above)</i></span>"
                    "</div>")
        if not ann.get("gaps"):
            st.caption("Model reported no gaps.")

    with t_tree:
        depth = st.slider("Expand depth", 1, 6, 2, key="tree_depth")
        st.json(rec, expanded=depth)

    with t_raw:
        st.code(json.dumps(rec, indent=2, ensure_ascii=False), language="json")

with right:
    prev = mine.get(tid, {})
    if prev:
        st.success(f"You reviewed this on {prev.get('annotated_at', '')[:16].replace('T', ' ')} UTC. "
                   "Saving again updates it.")

    st.subheader("Your review")
    taxonomy = sort_tree(load_taxonomy())
    cat_ids = [n["id"] for n in taxonomy]

    gap_options = ["True", "False"]
    gap_default = None if "gap_exists" not in prev else gap_options.index(str(prev["gap_exists"]))
    gap_exists = st.radio("1 · Does a comprehension gap exist?", gap_options,
                          index=gap_default, horizontal=True, key=f"gap_{tid}")

    cats_key = f"cats_{tid}"
    if cats_key not in st.session_state:
        st.session_state[cats_key] = prev.get("gap_categories", [])
    # Drop anything a teammate deleted meanwhile, or the widget would error.
    st.session_state[cats_key] = [c for c in st.session_state[cats_key] if c in cat_ids]
    st.multiselect("2 · Gap category (main › sub › …)", cat_ids, key=cats_key,
                   format_func=pretty, placeholder="Pick one or more categories…",
                   help="Pick the most specific category that fits. Missing one? Add it below.")

    TOP = "➕ (new main category)"

    def on_add_category() -> None:
        name = st.session_state.get("new_cat_name", "").strip()
        if not name:
            st.session_state["_flash"] = ("error", "Give the new category a name.")
            return
        parent = st.session_state.get("new_cat_parent")
        parent = "" if parent == TOP else parent
        cid = add_category(parent, name, st.session_state.get("new_cat_desc", ""), annotator)
        if st.session_state.get("new_cat_tag", True) and cid not in st.session_state[cats_key]:
            st.session_state[cats_key] = st.session_state[cats_key] + [cid]
        st.session_state["_flash"] = ("success", f"Added category {pretty(cid)}.")

    with st.expander("➕ Add a category or sub-category"):
        with st.form("add_cat", clear_on_submit=True, border=False):
            st.selectbox("Put it under", [TOP] + cat_ids, key="new_cat_parent",
                         format_func=lambda c: c if c == TOP else pretty(c))
            st.text_input("Name", key="new_cat_name", placeholder="e.g. edge_case_missed")
            st.text_area("Description (what misunderstanding it captures)", key="new_cat_desc",
                         height=80)
            st.checkbox("Also tag this thread with it", value=True, key="new_cat_tag")
            st.form_submit_button("Add category", on_click=on_add_category)

    agree = st.radio("Do you agree with the model's annotation?",
                     ["Agree", "Partially", "Disagree"],
                     index=["Agree", "Partially", "Disagree"].index(prev["model_agreement"])
                     if prev.get("model_agreement") else None,
                     horizontal=True, key=f"agree_{tid}")

    notes = st.text_area("Notes", value=prev.get("notes", ""), height=140, key=f"notes_{tid}")

    def save(advance: bool) -> None:
        if st.session_state.get(f"gap_{tid}") is None:
            st.session_state["_flash"] = ("error", "Choose True or False for question 1 first.")
            return
        cats = list(st.session_state.get(cats_key, []))
        review = {
            "annotator": annotator,
            "gap_exists": st.session_state[f"gap_{tid}"] == "True",
            "gap_categories": cats,
            "gap_main_categories": sorted({c.split(SEP)[0] for c in cats}),
            "model_agreement": st.session_state.get(f"agree_{tid}"),
            "notes": st.session_state.get(f"notes_{tid}", ""),
            "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        upsert_annotation({**rec, "human_review": review})
        st.session_state["_flash"] = ("success", f"Saved thread {tid}.")
        # With the "Not reviewed by me" filter the saved thread drops out of the
        # list, so the next one slides into the current position by itself.
        if advance and show != "Not reviewed by me":
            go(1)

    b1, b2 = st.columns(2)
    b1.button("💾 Save", on_click=save, args=(False,), use_container_width=True)
    b2.button("💾 Save & next ▶", on_click=save, args=(True,), type="primary",
              use_container_width=True)

    flash = st.session_state.pop("_flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])


# ------------------------------------------------- live taxonomy (all reviewers)

@st.fragment(run_every="5s")
def taxonomy_panel() -> None:
    """Re-reads the shared taxonomy every few seconds, so categories added by
    anyone on the team show up here without a page reload."""
    nodes = sort_tree(load_taxonomy())
    counts = usage_counts(load_annotations())
    new = [n for n in nodes if n.get("created_by") != "base"]

    st.subheader("🗂️ Gap category taxonomy (live)")
    st.caption(f"{len(nodes)} categories · {len(new)} added by the team · "
               f"numbers = reviews using the category (incl. its sub-categories) · "
               f"refreshes every 5 s")

    t_tree, t_graph, t_new = st.tabs(["🌳 Tree", "🕸️ Graph", f"🆕 Team-added ({len(new)})"])

    with t_tree:
        lines = []
        for n in nodes:
            depth = n["id"].count(SEP)
            name = n["id"].split(SEP)[-1]
            badge = "" if n.get("created_by") == "base" else f" 🆕 _by {n['created_by']}_"
            desc = f" — {n['description']}" if n.get("description") else ""
            lines.append(f"{'    ' * depth}- **{name}** `{counts.get(n['id'], 0)}`{badge}{desc}")
            if n.get("examples"):
                lines.append(f"{'    ' * (depth + 1)}- _e.g. {n['examples']}_")
        st.markdown("\n".join(lines))

    with t_graph:
        dot = ['digraph G { rankdir=LR; node [shape=box, style="rounded,filled", '
               'fontname="Helvetica", fontsize=11];', '"ROOT" [label="Gap categories", fillcolor="#e5e7eb"];']
        for n in nodes:
            cid = n["id"]
            color = "#c7d2fe" if n.get("created_by") == "base" else "#bbf7d0"
            label = f"{cid.split(SEP)[-1]}\\n({counts.get(cid, 0)})"
            dot.append(f'"{cid}" [label="{label}", fillcolor="{color}"];')
            parent = cid.rsplit(SEP, 1)[0] if SEP in cid else "ROOT"
            dot.append(f'"{parent}" -> "{cid}";')
        dot.append("}")
        st.graphviz_chart("\n".join(dot), use_container_width=True)
        st.caption("🟦 base category · 🟩 added by the team")

    with t_new:
        if not new:
            st.caption("No team-added categories yet.")
        for n in sorted(new, key=lambda n: n["created_at"], reverse=True):
            used = counts.get(n["id"], 0)
            has_children = any(m["id"].startswith(n["id"] + SEP) for m in nodes)
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{pretty(n['id'])}** · by {n['created_by']} · "
                        f"{n['created_at'][:16].replace('T', ' ')} · used {used}×"
                        + (f"  \n{n['description']}" if n.get("description") else ""))
            if used == 0 and not has_children:
                if c2.button("🗑️", key=f"del_{n['id']}", help="Delete (only unused categories)"):
                    delete_category(n["id"])
                    st.rerun()


st.divider()
taxonomy_panel()
