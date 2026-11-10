# Gap Annotator

A small Streamlit app for manually reviewing comprehension-gap annotations
stored as JSONL (one thread per line). Several team members can use it at the
same time; every review is saved to one shared JSONL file on the server.

## What each reviewer does

1. Enter your name in the sidebar.
2. Step through threads (Prev / Next, "Jump to", or filter "Not reviewed by me").
3. Read the thread in the **Readable**, **JSON tree** (collapsible) or **Raw** tab.
4. Answer:
   - **Does a comprehension gap exist?** True / False
   - **Gap category**: pick one or more nodes from the shared hierarchical
     taxonomy (main › sub › sub-sub …). Missing one? "Add a category or
     sub-category" puts it under any existing node (or as a new main category)
     and tags the thread with it.
   - Agreement with the model, plus free notes.
5. **Save** or **Save & next**.

The bottom of the page shows the **live taxonomy** (tree, graph, and a list of
team-added categories with who added them). It refreshes every 5 seconds, so
categories added by teammates appear without reloading. Each node shows how many
reviews use it (sub-category uses count toward the parent). Unused team-added
categories can be deleted there.

### Base main categories

Seeded on first run into `data/taxonomy.json` (edit `BASE_CATEGORIES` in `app.py`
before the first run to change them):

| Category | Question |
|---|---|
| `CODE_BEHAVIOR` | How the submitted code executes or what result it produces (incl. edge cases) |
| `SCOPE_AND_CONTEXT` | What was in the patch / already in the system / which parts it affected |
| `INTENT_AND_RATIONALE` | The goal of the change or why the implementation was chosen (requirements) |
| `VALIDATION` | What a test or check establishes |

## Output

`data/annotations.jsonl`: one line per *(thread_id, annotator)*. Each line is the
original record plus:

```json
"human_review": {
  "annotator": "lara",
  "gap_exists": true,
  "gap_categories": ["VALIDATION/test_scope_misread"],
  "gap_main_categories": ["VALIDATION"],
  "model_agreement": "Agree",
  "notes": "...",
  "annotated_at": "2026-11-06T10:15:00+00:00"
}
```

Saving a thread again replaces your earlier line, so two people reviewing the
same thread give two lines (handy for inter-annotator agreement). Writes are
file-locked, so simultaneous saves don't clobber each other.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Put your input file at `data/input.jsonl`, or upload it from the sidebar
("Use as server input file for the team" makes it the shared input).

Paths can be changed with env vars: `INPUT_PATH`, `OUTPUT_PATH`, `TAXONOMY_PATH`, `DATA_DIR`.

## Deploy on your server (Docker)

```bash
docker build -t gap-annotator .
docker run -d --name gap-annotator -p 8501:8501 -v $(pwd)/data:/app/data gap-annotator
```

Then open `http://<server>:8501`. The app has no login; put it behind your
VPN or a reverse proxy with basic auth if the server is public.

## Streamlit Community Cloud?

It works, but its disk is wiped on every restart, so annotations would be lost.
Use your own server (above) for real annotation work.
