from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.model_runtime import DEFAULT_LOW_CONF, predict_batch  # type: ignore


def _read_texts(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[str]:
    texts: List[str] = []

    if args.text:
        texts.extend(t.strip() for t in args.text if t and t.strip())

    if args.file:
        fpath = Path(args.file)
        if not fpath.exists():
            parser.error(f"File not found: {fpath}")
        with fpath.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    texts.append(line)

    if args.csv:
        cpath = Path(args.csv)
        if not cpath.exists():
            parser.error(f"CSV file not found: {cpath}")
        with cpath.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or args.column not in reader.fieldnames:
                parser.error(
                    f"Column '{args.column}' not found; available: {reader.fieldnames or 'none'}"
                )
            for row in reader:
                val = row.get(args.column, "") or ""
                val = val.strip()
                if val:
                    texts.append(val)

    if args.stdin:
        for line in sys.stdin:
            line = line.strip()
            if line:
                texts.append(line)

    if not texts:
        parser.error("Provide text via --text/--file/--csv or --stdin.")
    return texts


def _read_file_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    lines: List[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def _read_csv_column(path: Path, column: str) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' not found; available: {reader.fieldnames or 'none'}")
        values: List[str] = []
        for row in reader:
            val = row.get(column, "") or ""
            val = val.strip()
            if val:
                values.append(val)
        return values


def _shorten(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _print_table(preds: List[dict], text_limit: int | None) -> None:
    headers = ["text", "label", "conf", "needs_review", "coarse", "coarse_conf"]
    rows = []
    for p in preds:
        rows.append(
            [
                _shorten(str(p.get("text", "")), text_limit),
                str(p.get("label", "")),
                f"{float(p.get('confidence', 0.0)):.3f}",
                str(bool(p.get("needs_review", False))),
                str(p.get("coarse_label", "")),
                f"{float(p.get('coarse_confidence', 0.0)):.3f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(line: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(line))

    print(render(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(render(row))


def _emit_predictions(preds: List[dict], json_mode: bool, truncate: bool) -> None:
    if json_mode:
        for p in preds:
            print(json.dumps(p, ensure_ascii=True))
    else:
        text_limit = 96 if truncate else None
        _print_table(preds, text_limit)


def _interactive_loop(low_conf: float, json_mode: bool, truncate: bool) -> None:
    print(
        "Interactive TXCAT CLI. Enter a transaction line to score, or commands:\n"
        "  :help                 show this help\n"
        "  :q | :quit | :exit    quit\n"
        "  :json on|off          toggle JSON output (currently {json})\n"
        "  :low <float>          set low-confidence threshold (currently {low})\n"
        "  :file <path>          score newline-delimited text file\n"
        "  :csv <path> <column>  score a CSV column\n"
        "  :truncate on|off      toggle text truncation in table output (currently {truncate})\n"
        .format(json="on" if json_mode else "off", low=low_conf, truncate="on" if truncate else "off")
    )

    current_json = json_mode
    current_low = low_conf
    current_truncate = truncate

    while True:
        try:
            line = input("txcat> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in {":q", ":quit", ":exit"}:
            break
        if line == ":help":
            print(
                "Commands:\n"
                "  :q | :quit | :exit           quit\n"
                "  :json on|off                 toggle JSON output\n"
                "  :low <float>                 set low-confidence threshold\n"
                "  :file <path>                 score newline-delimited text file\n"
                "  :csv <path> <column>         score a CSV column\n"
                "  :truncate on|off             toggle text truncation in table output\n"
                "  <any other text>             score a single transaction line"
            )
            continue

        if line.startswith(":json"):
            parts = line.split()
            if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
                current_json = parts[1].lower() == "on"
                print(f"JSON output {'enabled' if current_json else 'disabled'}.")
            else:
                print("Usage: :json on|off")
            continue

        if line.startswith(":truncate"):
            parts = line.split()
            if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
                current_truncate = parts[1].lower() == "on"
                print(f"Truncation {'enabled' if current_truncate else 'disabled'}.")
            else:
                print("Usage: :truncate on|off")
            continue

        if line.startswith(":low"):
            parts = line.split()
            if len(parts) == 2:
                try:
                    current_low = float(parts[1])
                    print(f"Low-confidence threshold set to {current_low}.")
                except ValueError:
                    print("Usage: :low <float>")
            else:
                print("Usage: :low <float>")
            continue

        if line.startswith(":file"):
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print("Usage: :file <path>")
                continue
            try:
                texts = _read_file_lines(Path(parts[1]))
                preds = predict_batch(texts, low_conf=current_low)
                _emit_predictions(preds, current_json, current_truncate)
            except Exception as exc:
                print(f"Error: {exc}")
            continue

        if line.startswith(":csv"):
            parts = line.split()
            if len(parts) != 3:
                print("Usage: :csv <path> <column>")
                continue
            try:
                texts = _read_csv_column(Path(parts[1]), parts[2])
                preds = predict_batch(texts, low_conf=current_low)
                _emit_predictions(preds, current_json, current_truncate)
            except Exception as exc:
                print(f"Error: {exc}")
            continue

        # Default: treat as a single text to score
        preds = predict_batch([line], low_conf=current_low)
        _emit_predictions(preds, current_json, current_truncate)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TXCAT transaction categorisation from the CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--text",
        action="append",
        help="Text to score (repeatable).",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Path to a newline-delimited text file to score.",
    )
    parser.add_argument(
        "--csv",
        dest="csv",
        help="CSV file to read texts from.",
    )
    parser.add_argument(
        "--column",
        default="narrative",
        help="CSV column name containing the text to score.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read newline-delimited texts from stdin as well.",
    )
    parser.add_argument(
        "--low-conf",
        type=float,
        default=DEFAULT_LOW_CONF,
        help="Review threshold; predictions below this are flagged for review.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON lines instead of a table.",
    )
    parser.add_argument(
        "--no-truncate",
        dest="truncate",
        action="store_false",
        help="Do not shorten long text fields in table output.",
    )
    parser.set_defaults(truncate=True)

    args = parser.parse_args()
    if args.text or args.file or args.csv or args.stdin:
        texts = _read_texts(args, parser)
        preds = predict_batch(texts, low_conf=args.low_conf)
        _emit_predictions(preds, args.json, args.truncate)
    else:
        _interactive_loop(low_conf=args.low_conf, json_mode=args.json, truncate=args.truncate)


if __name__ == "__main__":
    main()
