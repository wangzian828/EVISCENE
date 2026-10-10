# -*- coding: utf-8 -*-
"""CLI for rendering one or more trace JSON files into HTML viewers."""
from __future__ import annotations

import argparse
import os
from typing import Iterable, List

from eviscene.trace_viewer import render_trace_file


def _iter_trace_files(path: str) -> Iterable[str]:
    if os.path.isfile(path):
        if path.lower().endswith(".json"):
            yield path
        return

    for root, _, files in os.walk(path):
        for name in files:
            if name.lower().endswith(".json"):
                yield os.path.join(root, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render trace JSON files to standalone HTML viewers")
    parser.add_argument("input_path", help="A trace JSON file or a directory containing trace JSON files")
    parser.add_argument("--output-dir", default="", help="Optional output directory for generated HTML files")
    args = parser.parse_args()

    trace_files: List[str] = list(_iter_trace_files(args.input_path))
    if not trace_files:
        raise SystemExit(f"No trace JSON files found under: {args.input_path}")

    generated = 0
    for json_path in trace_files:
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            html_name = os.path.splitext(os.path.basename(json_path))[0] + ".html"
            html_path = os.path.join(args.output_dir, html_name)
        else:
            html_path = os.path.splitext(json_path)[0] + ".html"
        render_trace_file(json_path, html_path)
        print(html_path)
        generated += 1

    print(f"Rendered {generated} HTML file(s).")


if __name__ == "__main__":
    main()
