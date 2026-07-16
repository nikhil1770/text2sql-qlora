"""
Text-to-SQL demo — fine-tuned Qwen2.5-Coder-3B (QLoRA) on the Olist e-commerce DB.

Run (from the repo root, GPU required — Colab T4 works):
    streamlit run app/app.py

Secrets / keys — NEVER hardcode a token in this file:
    * The adapter repo on HF Hub is public, so normally NO token is needed.
    * If you ever need one (private repo / rate limits), set it as an
      environment variable before launching:
          export HF_TOKEN=hf_xxx            # local terminal
      or store it in Colab Secrets (the key icon in the left sidebar) under
      the name HF_TOKEN — this app picks up both automatically.
"""

import os
import sys
import time
import sqlite3

import pandas as pd
import streamlit as st

# Make `src/` importable when launched as `streamlit run app/app.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.guardrails import is_safe_sql, enforce_limit, DEFAULT_ROW_LIMIT

# ---------------------------------------------------------------- config ----

ADAPTER_REPO = "nikhil1772000/text2sql-qlora-qwen2.5-coder-3b"

# First existing path wins; override with env var OLIST_DB_PATH if needed.
DB_CANDIDATES = [
    os.environ.get("OLIST_DB_PATH", ""),
    os.path.join(REPO_ROOT, "data", "olist.sqlite"),
    os.path.join(REPO_ROOT, "olist_db", "olist.sqlite"),
    "olist_db/olist.sqlite",
    "olist.sqlite",
]

# EXACTLY the system prompt used in training/eval — do not edit, or the
# prompt distribution shifts away from what the adapter was trained on.
SYSTEM_PROMPT = (
    "You are an expert data analyst who writes SQLite SQL queries. "
    "Given a database schema and a natural-language question, respond with a "
    "single valid SQL query that answers it. Use only the tables and columns "
    "in the schema. Output only the SQL query, with no explanation."
)

EXAMPLE_QUESTIONS = [
    "How many orders were delivered in 2017?",
    "What are the top 5 product categories by number of orders?",
    "What is the average review score per seller state?",
    "Which city has the most customers?",
    "What is the total payment value per payment type?",
]


def get_hf_token():
    """Env var first (local), then Colab Secrets. Returns None if absent —
    fine for a public adapter repo."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from google.colab import userdata  # only exists inside Colab
        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def find_db_path():
    for p in DB_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


# ------------------------------------------------------- cached resources ----

@st.cache_resource(show_spinner=False)
def load_model():
    """Load base Qwen2.5-Coder-3B in 4-bit and attach the LoRA adapter.
    Cached so it loads once per Streamlit session, not on every rerun."""
    token = get_hf_token()
    if token:
        from huggingface_hub import login
        login(token=token)

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_REPO,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_schema(db_path: str):
    """Read the live schema from SQLite and return:
    (a) a Spider-style one-line string for the model prompt — the same format
        the model saw in training, and
    (b) a {table: [(col, type), ...]} dict for the sidebar explorer."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = [r[0] for r in cur.fetchall()]

    table_cols = {}
    parts = []
    for t in tables:
        cur.execute(f'PRAGMA table_info("{t}")')          # app-side only; user
        cols = [(r[1], r[2] or "TEXT") for r in cur.fetchall()]  # SQL can't PRAGMA
        table_cols[t] = cols
        col_str = " , ".join(f"{c} ({ty})" for c, ty in cols)
        parts.append(f"{t} : {col_str}")
    conn.close()

    schema_string = "Tables and columns:\n" + " | ".join(parts)
    return schema_string, table_cols


# ----------------------------------------------------------- core actions ----

def generate_sql_with_retry(model, tokenizer, schema_string, question,
                            db_path, max_retries=2):
    """Generate SQL; if it fails to execute, show the model the error and
    let it repair itself (same technique as evaluation.ipynb, +1.7 pts)."""
    sql = generate_sql(model, tokenizer, schema_string, question)
    for attempt in range(max_retries):
        ok, _ = is_safe_sql(sql)
        if not ok:
            return sql, attempt            # let the UI show the block badge
        try:
            run_readonly(db_path, enforce_limit(sql))
            return sql, attempt            # executes fine — done
        except Exception as e:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Database schema:\n{schema_string}\n\nQuestion: {question}"},
                {"role": "assistant", "content": sql},
                {"role": "user", "content":
                    f"That query failed with this SQLite error:\n{e}\n\n"
                    "Write a corrected SQL query. Output only the SQL, "
                    "no explanation."},
            ]
            enc = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True,
                return_tensors="pt", return_dict=True).to(model.device)
            out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][enc["input_ids"].shape[-1]:],
                                    skip_special_tokens=True)
            sql = text.replace("```sql", "").replace("```", "").strip().split(";")[0].strip()
    return sql, max_retries

def run_readonly(db_path: str, sql: str) -> pd.DataFrame:
    """Layer 2 guardrail: mode=ro means the engine itself refuses writes."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def maybe_chart(df: pd.DataFrame):
    """Auto-chart when the shape suits it: one label column + one numeric
    column, and few enough rows to read as a bar chart."""
    if df.shape[1] == 2 and 1 < len(df) <= 25:
        label, value = df.columns[0], df.columns[1]
        if pd.api.types.is_numeric_dtype(df[value]):
            st.bar_chart(df.set_index(label)[value])


# -------------------------------------------------------------------- UI ----

st.set_page_config(
    page_title="Text-to-SQL · Olist", page_icon="🗃️", layout="wide"
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; }
      .app-title { font-size: 1.9rem; font-weight: 700; margin-bottom: 0; }
      .app-sub   { color: #8b8b95; margin-top: 0.15rem; font-size: 0.95rem; }
      .badge-ok, .badge-block {
        display: inline-block; padding: 2px 12px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600;
      }
      .badge-ok    { background: #123524; color: #7ee2a8; }
      .badge-block { background: #3a1220; color: #ff8fa3; }
      div[data-testid="stCodeBlock"] { font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="app-title">🗃️ Ask the Olist database</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Natural language → SQL, generated by '
    'Qwen2.5-Coder-3B fine-tuned with QLoRA on Spider · executed read-only '
    'with guardrails</p>',
    unsafe_allow_html=True,
)

db_path = find_db_path()
if db_path is None:
    st.error(
        "olist.sqlite not found. Put it at `data/olist.sqlite` or set the "
        "`OLIST_DB_PATH` environment variable."
    )
    st.stop()

schema_string, table_cols = load_schema(db_path)

# ---- sidebar: model card + schema explorer ----
with st.sidebar:
    st.subheader("Model")
    st.markdown(
        f"**Adapter:** [`{ADAPTER_REPO.split('/')[-1]}`]"
        f"(https://huggingface.co/{ADAPTER_REPO})\n\n"
        "**Base:** Qwen2.5-Coder-3B-Instruct (4-bit)\n\n"
        "**Eval:** 72.3% execution accuracy · Spider dev (n=300)"
    )
    st.divider()
    st.subheader("Database schema")
    st.caption("Olist Brazilian e-commerce · unseen by the model in training")
    for t, cols in table_cols.items():
        with st.expander(f"{t}  ·  {len(cols)} cols"):
            st.table(pd.DataFrame(cols, columns=["column", "type"]))
    st.divider()
    st.caption(
        "Guardrails: single SELECT/WITH only · no comments · "
        "keyword blocklist · read-only connection · "
        f"LIMIT {DEFAULT_ROW_LIMIT} cap"
    )

# ---- example question buttons ----
st.write("**Try one:**")
btn_cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, q in zip(btn_cols, EXAMPLE_QUESTIONS):
    if col.button(q, use_container_width=True):
        st.session_state["question"] = q

question = st.text_input(
    "Your question about the data",
    key="question",
    placeholder="e.g. Which product category has the highest average price?",
)

go = st.button("Generate SQL & run", type="primary", disabled=not question)

if go and question:
    with st.spinner("Loading model (first time only) …"):
        try:
            model, tokenizer = load_model()
        except Exception as e:
            st.error(
                "Model failed to load — this app needs an NVIDIA GPU "
                "(Colab T4 works). If the HF repo is private, set HF_TOKEN "
                f"as described in the file header.\n\nDetails: {e}"
            )
            st.stop()

    with st.spinner("Generating SQL …"):
        t0 = time.time()
        sql, n_retries = generate_sql_with_retry(
            model, tokenizer, schema_string, question, db_path)
        gen_s = time.time() - t0

    caption = f"**Generated SQL** · {gen_s:.1f}s"
    if n_retries:
        caption += f" · self-corrected ×{n_retries}"
    st.write(caption)
    st.code(sql, language="sql")

    ok, reason = is_safe_sql(sql)
    if not ok:
        st.markdown(
            f'<span class="badge-block">🛑 Blocked by guardrails</span> '
            f'&nbsp;{reason}',
            unsafe_allow_html=True,
        )
        st.info(
            "The query was generated but not executed. Only single, "
            "read-only SELECT statements are allowed."
        )
    else:
        st.markdown(
            '<span class="badge-ok">✓ Passed guardrails</span>',
            unsafe_allow_html=True,
        )
        safe_sql = enforce_limit(sql)
        try:
            t0 = time.time()
            df = run_readonly(db_path, safe_sql)
            run_s = time.time() - t0
        except Exception as e:
            st.error(f"Query passed guardrails but failed to execute: {e}")
            st.stop()

        st.write(f"**Results** · {len(df)} rows · {run_s:.2f}s")
        st.dataframe(df, use_container_width=True)
        maybe_chart(df)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode(),
            file_name="query_result.csv",
            mime="text/csv",
        )
