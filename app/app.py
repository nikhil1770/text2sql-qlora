"""
Text-to-SQL Playground (QLoRA fine-tuned Qwen2.5-Coder-3B)
==========================================================
A Streamlit app that turns plain-English questions into SQL for the Olist
e-commerce database, using a QLoRA fine-tuned model. Compare the base model
vs the fine-tuned model, adjust decoding, and run the generated SQL live.

Run (in Colab, GPU on):
    !streamlit run app/app.py & npx localtunnel --port 8501
"""

import os
import re
import sqlite3

import pandas as pd
import streamlit as st

# --- Guardrails: import the shared safety check from src/ -------------------
# We add the repo root to the path so `from src.guardrails import ...` works
# regardless of where streamlit is launched from.
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.guardrails import is_safe_sql
except Exception:
    # Fallback so the app still runs if src/ isn't found; keeps the demo alive.
    def is_safe_sql(query: str):
        q = (query or "").strip().upper()
        if not (q.startswith("SELECT") or q.startswith("WITH")):
            return False, "Only SELECT queries are allowed."
        return True, "OK"


# ============================================================================
# CONFIG
# ============================================================================
DB_PATH = os.environ.get("OLIST_DB_PATH", "olist_db/olist.sqlite")
ADAPTER_REPO = "nikhil1772000/text2sql-qlora-qwen2.5-coder-3b"

SYSTEM_PROMPT = (
    "You are an expert data analyst who writes SQLite SQL queries. "
    "Given a database schema and a natural-language question, respond with a "
    "single valid SQL query that answers it. Use only the tables and columns "
    "in the schema. Output only the SQL query, with no explanation."
)

# Example questions to help users start (empty-state guidance).
EXAMPLE_QUESTIONS = [
    "How many orders are there in total?",
    "What are the top 5 product categories by number of orders?",
    "What is the average review score?",
    "Which cities have the most customers?",
    "What is the total payment value per payment type?",
]


# ============================================================================
# DATABASE HELPERS  (no GPU needed — safe to build/test on CPU)
# ============================================================================
@st.cache_resource
def get_schema_string(db_path: str) -> str:
    """Read the live schema from the SQLite file as a Spider-style string."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    parts = []
    for t in tables:
        cur.execute(f"PRAGMA table_info('{t}');")
        cols = cur.fetchall()
        col_strs = [f"{c[1]} ({c[2] or 'text'})" for c in cols]
        parts.append(f"{t} : {' , '.join(col_strs)}")
    conn.close()
    return " | ".join(parts)


def run_sql(query: str, db_path: str):
    """Run a read-only query. Returns (dataframe, error_message)."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)


def clean_sql(text: str) -> str:
    """Strip markdown fences and keep only the first statement."""
    text = re.sub(r"```sql|```", "", text).strip()
    return text.split(";")[0].strip()


# ============================================================================
# MODEL  (GPU needed — filled in on the GPU pass; stubbed so UI runs on CPU)
# ============================================================================
@st.cache_resource(show_spinner="Loading the model (first run only)...")
def load_model():
    """
    Load base Qwen + the fine-tuned adapter once, cached across reruns.
    Returns (model, tokenizer). Requires a GPU.
    """
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_REPO,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_sql(model, tokenizer, schema, question, use_finetuned,
                 temperature, top_p, max_new_tokens=256):
    """Generate SQL. Toggle adapter off (base) or on (fine-tuned)."""
    user = f"Database schema:\n{schema}\n\nQuestion: {question}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)

    # Greedy when temperature ~ 0, else sample with the given temp/top_p.
    do_sample = temperature > 0.01
    gen_kwargs = dict(max_new_tokens=max_new_tokens,
                      pad_token_id=tokenizer.eos_token_id, do_sample=do_sample)
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    from contextlib import nullcontext
    ctx = model.disable_adapter() if not use_finetuned else nullcontext()
    with ctx:
        out = model.generate(**enc, **gen_kwargs)

    gen = out[0][enc["input_ids"].shape[-1]:]
    return clean_sql(tokenizer.decode(gen, skip_special_tokens=True))


# ============================================================================
# UI
# ============================================================================
st.set_page_config(page_title="Text-to-SQL Playground", page_icon="🗄️",
                   layout="wide")

st.title("Text-to-SQL Playground")
st.caption("Ask a question in plain English — get SQL for the Olist "
           "e-commerce database. Fine-tuned Qwen2.5-Coder-3B (QLoRA).")

# --- Sidebar: controls -----------------------------------------------------
with st.sidebar:
    st.header("Model")
    use_finetuned = st.toggle("Use fine-tuned model", value=True,
                              help="Off = base Qwen. On = your QLoRA adapter.")

    st.selectbox(
        "LoRA rank",
        options=["r16 (active)", "r8 (coming soon)", "r32 (coming soon)"],
        index=0,
        help="Only rank 16 is trained so far. Others are placeholders.",
    )

    st.header("Decoding")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.05,
                            help="0 = deterministic (greedy). Higher = more random.")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05,
                      help="Nucleus sampling threshold (used when temperature > 0).")

    st.divider()
    st.caption("Data: Olist Brazilian e-commerce (CC BY-NC 4.0). "
               "Read-only: only SELECT queries run.")

# --- Load schema (works on CPU) --------------------------------------------
db_exists = os.path.exists(DB_PATH)
if db_exists:
    schema = get_schema_string(DB_PATH)
else:
    schema = ""
    st.warning(f"Database not found at `{DB_PATH}`. "
               "Download the Olist SQLite file before running queries.")

with st.expander("View database schema"):
    if schema:
        for tbl in schema.split(" | "):
            st.text(tbl)
    else:
        st.text("Schema unavailable — database file missing.")

# --- Question input ---------------------------------------------------------
st.subheader("Ask a question")
example = st.selectbox("Pick an example, or write your own below:",
                       ["(write my own)"] + EXAMPLE_QUESTIONS)
default_q = "" if example == "(write my own)" else example
question = st.text_input("Your question", value=default_q,
                         placeholder="e.g. How many orders were delivered?")

run = st.button("Generate SQL", type="primary")

# --- Action -----------------------------------------------------------------
if run:
    if not question.strip():
        st.error("Type a question first.")
    elif not db_exists:
        st.error("No database available to query.")
    else:
        with st.spinner("Generating..."):
            try:
                model, tokenizer = load_model()
                sql = generate_sql(model, tokenizer, schema, question,
                                   use_finetuned, temperature, top_p)
            except Exception as e:
                sql = None
                st.error(f"Model unavailable: {e}\n\n"
                         "If you're on CPU or the GPU quota is exhausted, "
                         "generation won't run. Try again on a GPU runtime.")

        if sql:
            model_label = "Fine-tuned" if use_finetuned else "Base"
            st.markdown(f"**Generated SQL** ({model_label} model):")
            st.code(sql, language="sql")

            # Guardrail check BEFORE running anything
            safe, reason = is_safe_sql(sql)
            if not safe:
                st.error(f"Query blocked by guardrails: {reason}")
            else:
                df, err = run_sql(sql, DB_PATH)
                if err:
                    st.warning(f"SQL ran but returned an error: {err}")
                elif df is None or df.empty:
                    st.info("Query ran successfully but returned no rows.")
                else:
                    st.success(f"Returned {len(df)} row(s).")
                    st.dataframe(df, use_container_width=True)
