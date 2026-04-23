from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import get_settings
from data_access.discovery import default_search_topics, search_dataflows


def main() -> None:
    settings = get_settings()
    output_dir = settings.resolve_path(settings.eurostat_api.search_results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_topic: list[pd.DataFrame] = []
    markdown_lines = [
        "# Eurostat Discovery Search",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        "",
    ]

    for topic in default_search_topics():
        results = search_dataflows([topic]).head(settings.eurostat_api.default_search_limit).copy()
        markdown_lines.append(f"## {topic}")
        markdown_lines.append("")

        if results.empty:
            markdown_lines.append("_No matching dataflows found._")
            markdown_lines.append("")
            continue

        results.insert(0, "topic", topic)
        results_by_topic.append(results)

        table_rows = [
            (
                row.dataset_id,
                row.title,
                str(row.match_score),
                row.matched_keywords,
            )
            for row in results.itertuples(index=False)
        ]
        markdown_lines.append(
            _markdown_table(
                ("dataset_id", "title", "match_score", "matched_keywords"),
                table_rows,
            )
        )
        markdown_lines.append("")

    combined = (
        pd.concat(results_by_topic, ignore_index=True)
        if results_by_topic
        else pd.DataFrame(columns=["topic", "dataset_id", "title", "match_score", "matched_keywords"])
    )
    csv_path = output_dir / "eurostat_topic_search.csv"
    markdown_path = output_dir / "eurostat_topic_search.md"

    combined.to_csv(csv_path, index=False)
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return "_No rows available._"

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body])


if __name__ == "__main__":
    main()
