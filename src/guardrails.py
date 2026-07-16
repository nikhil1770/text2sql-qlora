"""
SQL guardrails for the Text-to-SQL demo app.

Layer 1 of a defense-in-depth setup:
  Layer 1 (this file): reject anything that isn't a single, read-only
           SELECT/WITH statement, before it ever touches the database.
  Layer 2 (app.py):    the SQLite connection itself is opened read-only
           (mode=ro), so even if a bad query slips past the regex,
           writes fail at the database engine level.
"""

import re

# Anything that mutates data/schema, or escapes the sandbox (ATTACH lets you
# open other database files; PRAGMA can change engine behaviour).
FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "GRANT", "REVOKE",
    "PRAGMA", "ATTACH", "DETACH", "VACUUM", "REINDEX",
]

DEFAULT_ROW_LIMIT = 500


def is_safe_sql(query: str):
    """Return (ok: bool, reason: str). ok=True only for a single read-only
    SELECT/WITH statement with no comments and no forbidden keywords."""
    if not query or not query.strip():
        return False, "Empty query."

    q = query.strip()
    q_upper = q.upper()

    # Block comments — a classic way to hide a second statement or keyword.
    if "--" in q or "/*" in q or "*/" in q:
        return False, "SQL comments are not allowed."

    # Exactly one statement: a ';' is only OK as the final character.
    stripped = q.rstrip(";").strip()
    if ";" in stripped:
        return False, "Multiple SQL statements are not allowed."

    # Read-only entry points only.
    if not (q_upper.startswith("SELECT") or q_upper.startswith("WITH")):
        return False, "Only SELECT queries are allowed."

    # Whole-word keyword scan (word boundaries so e.g. a column named
    # 'created_at' does not false-positive on CREATE).
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", q_upper):
            return False, f"Forbidden operation detected: {kw}."

    return True, "OK"


def enforce_limit(query: str, max_rows: int = DEFAULT_ROW_LIMIT) -> str:
    """Append a LIMIT if the query doesn't already have one, so a broad
    SELECT can't return the entire table into the UI."""
    q = query.strip().rstrip(";").strip()
    if re.search(r"\bLIMIT\s+\d+", q, flags=re.IGNORECASE):
        return q
    return f"{q} LIMIT {max_rows}"
