"""Minimal output contract, separate from optional SQLGlot validation FEAT_16."""

from sqlglot.tokens import Tokenizer, TokenType
from sqlglot.errors import TokenError

VERSION = "query-output-v1"


def sql_output_issue(value):
    if not isinstance(value, str):
        return "non_text_sql"
    if not value.strip():
        return "empty_sql"
    try:
        tokens = Tokenizer(dialect="postgres").tokenize(value)
    except TokenError:
        return "invalid_sql_tokens"
    if not tokens:
        return "comments_only_sql"
    # This is an output-shape check, not a replacement for the FEAT_16 parser.
    kinds = [t.token_type for t in tokens]
    if TokenType.SEMICOLON in kinds[:-1]:
        return "multiple_statements"
    first = next((k for k in kinds if k != TokenType.L_PAREN), None)
    if first not in (TokenType.SELECT, TokenType.WITH, TokenType.VALUES):
        return "non_query_text"
    if any(
        k in kinds
        for k in (
            TokenType.DELETE,
            TokenType.UPDATE,
            TokenType.INSERT,
            TokenType.DROP,
            TokenType.CREATE,
            TokenType.ALTER,
            TokenType.COMMAND,
            TokenType.INTO,
        )
    ):
        return "non_read_query"
    if first == TokenType.WITH and not (
        TokenType.ALIAS in kinds
        and TokenType.L_PAREN in kinds
        and any(k in kinds for k in (TokenType.SELECT, TokenType.VALUES))
    ):
        return "non_query_with_text"
    return None


def guarded_response(response):
    """Keep valid SQL unchanged; invalid model answers become failures, never ambiguity TNs."""
    if not isinstance(response, dict) or response.get("status") not in (
        "success",
        "error",
        "ambiguous",
    ):
        return {"status": "error"}, "invalid_tool_response"
    if response["status"] != "success":
        return response, None
    issue = sql_output_issue(response.get("query"))
    return ({"status": "error"}, issue) if issue else (response, None)
