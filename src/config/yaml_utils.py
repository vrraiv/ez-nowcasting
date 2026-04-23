from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when PyYAML is unavailable.
    yaml = None


class MinimalYamlError(ValueError):
    """Raised when the lightweight YAML fallback cannot parse the document."""


@dataclass(frozen=True, slots=True)
class _YamlLine:
    indent: int
    content: str
    raw: str
    line_number: int


def load_yaml_document(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return _parse_minimal_yaml(text)


def _parse_minimal_yaml(text: str) -> Any:
    lines = _prepare_lines(text)
    if not lines:
        return {}
    value, next_index = _parse_node(lines, start_index=0, expected_indent=lines[0].indent)
    if next_index != len(lines):
        raise MinimalYamlError(f"Unexpected trailing content at line {lines[next_index].line_number}.")
    return value


def _prepare_lines(text: str) -> list[_YamlLine]:
    prepared: list[_YamlLine] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.rstrip()
        if not stripped.strip():
            continue
        content = stripped.lstrip(" ")
        if content.startswith("#"):
            continue
        indent = len(stripped) - len(content)
        prepared.append(_YamlLine(indent=indent, content=content, raw=stripped, line_number=line_number))
    return prepared


def _parse_node(lines: list[_YamlLine], start_index: int, expected_indent: int) -> tuple[Any, int]:
    if start_index >= len(lines):
        return {}, start_index

    current = lines[start_index]
    if current.indent < expected_indent:
        return {}, start_index

    if current.content.startswith("- "):
        return _parse_list(lines, start_index, current.indent)
    return _parse_mapping(lines, start_index, current.indent)


def _parse_mapping(lines: list[_YamlLine], start_index: int, expected_indent: int) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    index = start_index

    while index < len(lines):
        line = lines[index]
        if line.indent < expected_indent:
            break
        if line.indent != expected_indent:
            raise MinimalYamlError(
                f"Unexpected indentation at line {line.line_number}: expected {expected_indent} spaces."
            )
        if line.content.startswith("- "):
            break
        if ":" not in line.content:
            raise MinimalYamlError(f"Expected a mapping entry at line {line.line_number}.")

        key, raw_value = line.content.split(":", maxsplit=1)
        key = key.strip()
        value = raw_value.lstrip()

        if value in {">", "|"}:
            mapping[key], index = _parse_block_scalar(lines, index + 1, parent_indent=expected_indent, folded=value == ">")
            continue

        if value:
            mapping[key] = _parse_scalar(value)
            index += 1
            continue

        child, next_index = _parse_nested_value(lines, index + 1, expected_indent)
        mapping[key] = child
        index = next_index

    return mapping, index


def _parse_list(lines: list[_YamlLine], start_index: int, expected_indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    index = start_index

    while index < len(lines):
        line = lines[index]
        if line.indent < expected_indent:
            break
        if line.indent != expected_indent or not line.content.startswith("- "):
            break

        value = line.content[2:].lstrip()
        if value:
            items.append(_parse_scalar(value))
            index += 1
            continue

        child, next_index = _parse_nested_value(lines, index + 1, expected_indent)
        items.append(child)
        index = next_index

    return items, index


def _parse_nested_value(lines: list[_YamlLine], start_index: int, parent_indent: int) -> tuple[Any, int]:
    if start_index >= len(lines):
        return {}, start_index

    next_line = lines[start_index]
    if next_line.indent <= parent_indent:
        return {}, start_index

    return _parse_node(lines, start_index, next_line.indent)


def _parse_block_scalar(
    lines: list[_YamlLine],
    start_index: int,
    parent_indent: int,
    folded: bool,
) -> tuple[str, int]:
    if start_index >= len(lines) or lines[start_index].indent <= parent_indent:
        return "", start_index

    block_indent = lines[start_index].indent
    parts: list[str] = []
    index = start_index

    while index < len(lines):
        line = lines[index]
        if line.indent < block_indent:
            break
        parts.append(line.raw[block_indent:])
        index += 1

    if folded:
        collapsed = " ".join(part.strip() for part in parts if part.strip())
        return collapsed.strip(), index
    return "\n".join(parts).strip(), index


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]

    lowered = text.casefold()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None

    if text.isdigit():
        return int(text)

    try:
        return float(text)
    except ValueError:
        return text
