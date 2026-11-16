"""Strict taxonomy response contract: malformed output is never a negative verdict."""

import json
import re


class InvalidTaxonomy(ValueError):
    pass


def parse_taxonomy(text):
    if not isinstance(text, str):
        raise InvalidTaxonomy("Taxonomy response must be JSON text")
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.I | re.S)
    if match:
        text = match.group(1)

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise InvalidTaxonomy("Duplicate taxonomy field")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (ValueError, TypeError):
        raise InvalidTaxonomy("Malformed taxonomy JSON") from None
    if not isinstance(value, dict) or set(value) != {"is_ambiguous", "ambiguity_type", "reason"}:
        raise InvalidTaxonomy("Missing or unexpected taxonomy fields")
    if type(value["is_ambiguous"]) is not bool:
        raise InvalidTaxonomy("is_ambiguous must be a boolean")
    kind, reason = value["ambiguity_type"], value["reason"]
    if value["is_ambiguous"]:
        if not isinstance(kind, str) or kind not in {f"A{i}" for i in range(1, 7)}:
            raise InvalidTaxonomy("Ambiguous verdict requires an A1-A6 type")
    elif kind is not None:
        raise InvalidTaxonomy("Unambiguous verdict requires a null type")
    if reason is not None and not isinstance(reason, str):
        raise InvalidTaxonomy("reason must be text or null")
    return {**value, "reason": reason or ""}
