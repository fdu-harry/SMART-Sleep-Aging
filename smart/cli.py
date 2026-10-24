"""Render clinical narratives and prepare task prompts."""
import argparse
import json
from pathlib import Path

from .scn import render_scn
from .prompts import build_age_prompt, build_recommendation_prompt


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(prog="smart")
    subs = parser.add_subparsers(dest="command", required=True)
    scn = subs.add_parser("scn", help="Render a section/field JSON file into an SCN")
    scn.add_argument("--input", type=Path, required=True)
    scn.add_argument("--output", type=Path, required=True)
    age = subs.add_parser("age-prompt", help="Prepare an age-estimation prompt")
    age.add_argument("--input", type=Path, required=True)
    age.add_argument("--targets", type=Path, required=True)
    age.add_argument("--output", type=Path, required=True)
    recommendation = subs.add_parser("recommendation-prompt", help="Prepare a recommendation prompt")
    recommendation.add_argument("--input", type=Path, required=True)
    recommendation.add_argument("--table", type=Path, required=True)
    recommendation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "scn":
        text = render_scn(read_json(args.input))
    elif args.command == "age-prompt":
        text = build_age_prompt(read_json(args.targets), args.input.read_text(encoding="utf-8-sig"))
    else:
        text = build_recommendation_prompt(args.input.read_text(encoding="utf-8-sig"),
                                           args.table.read_text(encoding="utf-8-sig"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(text)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
