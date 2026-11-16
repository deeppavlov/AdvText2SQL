"""Remove SQL presentation wrappers without changing literals or query semantics."""

import re


def clean_sql(sql: str, dialect="postgres") -> str:
    if not isinstance(sql, str):
        raise ValueError("SQL must be text")
    sql = sql.strip()
    fence = re.fullmatch(r"```(?:sql|postgresql|postgres)?\s*\n?(.*?)\n?```", sql, re.I | re.S)
    if fence:
        sql = fence.group(1).strip()
    result = []
    i = 0
    while i < len(sql):
        if sql.startswith("--", i):
            end = sql.find("\n", i + 2)
            i = len(sql) if end < 0 else end
            result.append(" ")
        elif sql.startswith("/*", i):
            start, depth = i, 1
            i += 2
            while i < len(sql) and depth:
                if sql.startswith("/*", i):
                    depth += 1
                    i += 2
                elif sql.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise ValueError("Unterminated SQL comment")
            result.append("\n" * sql[start:i].count("\n") or " ")
        elif sql[i] in "'\"`" or (dialect == "sqlite" and sql[i] == "["):
            start, quote = i, "]" if sql[i] == "[" else sql[i]
            escaped = (quote == "'" and i > 0 and sql[i - 1] in "eE"
                       and (i < 2 or not (sql[i - 2].isalnum() or sql[i - 2] == "_")))
            i += 1
            while i < len(sql):
                if escaped and sql[i] == "\\":
                    i += 2
                elif sql[i] == quote:
                    i += 1
                    if i < len(sql) and sql[i] == quote:
                        i += 1
                    else:
                        break
                else:
                    i += 1
            else:
                raise ValueError("Unterminated SQL quoted value")
            result.append(sql[start:i])
        elif (sql[i] == "$" and (i == 0 or not (sql[i - 1].isalnum() or sql[i - 1] in "_$"))
              and (match := re.match(r"\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$", sql[i:]))):
            tag = match.group()
            end = sql.find(tag, i + len(tag))
            if end < 0:
                raise ValueError("Unterminated dollar-quoted value")
            end += len(tag)
            result.append(sql[i:end])
            i = end
        else:
            result.append(sql[i])
            i += 1
    return "".join(result).strip()
