"""
Guardrails for the Text-to-SQL app.
Inspects model-generated SQL before execution and blocks anything unsafe.
The app is READ-ONLY: only SELECT queries are permitted.
"""
import re

# Statement types that modify data or schema — never allowed in a read-only app
FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "TRUNCATE", "CREATE", "REPLACE", "GRANT", "REVOKE",
]


def is_safe_sql(query: str):
    """
    Check whether a SQL query is safe to execute (read-only).
    Returns (is_safe: bool, reason: str).
    """
    if not query or not query.strip():
        return False, "Empty query."

    q = query.strip()
    q_upper = q.upper()

    # 1) Block SQL comments (used to smuggle past filters)
    if "--" in q or "/*" in q or "*/" in q:
        return False, "SQL comments are not allowed."

    # 2) Block stacked/multiple statements (e.g. 'SELECT ...; DROP ...')
    #    A trailing semicolon is fine; a semicolon with more SQL after it is not.
    stripped = q.rstrip(";").strip()
    if ";" in stripped:
        return False, "Multiple SQL statements are not allowed."

    # 3) Must be a read-only SELECT (allow leading WITH ... SELECT / CTEs too)
    if not (q_upper.startswith("SELECT") or q_upper.startswith("WITH")):
        return False, "Only SELECT queries are allowed."

    # 4) Block any forbidden keyword appearing as a whole word
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", q_upper):
            return False, f"Forbidden operation detected: {kw}."

    return True, "OK"