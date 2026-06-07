#!/usr/bin/env python3
"""
ATT-1 M152: deterministic fuzz corpus coverage summary.

This is a static coverage guard for the local fuzz/smoke surface. It does not
execute fuzzers or inference; it counts the checked-in hostile JSON fixtures,
inline JSON mutation seeds, and C binary-loader smoke cases so release checks
can detect accidental corpus shrinkage.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_HOSTILE = _HERE / "fixtures" / "hostile"
_FUZZ_JSON = _HERE / "fuzz_json_schemas.py"
_FUZZ_LOADER = _REPO / "tests" / "fuzz_model_loader.c"

_MIN_HOSTILE_FIXTURES = 32
_MIN_JSON_MUTATIONS = 11
_MIN_LOADER_CASES = 22


def _schema_for_fixture(name: str) -> str:
    if name.startswith("placement_"):
        return "placement"
    if name.startswith("cmd_plan_"):
        return "command_plan"
    if name.startswith("route_"):
        return "fabric_route"
    if name.startswith("exec_"):
        return "execution_plan"
    if name.startswith("pipeline_"):
        return "pipeline"
    return "unknown"


def _count_json_mutations() -> int:
    text = _FUZZ_JSON.read_text(encoding="utf-8")
    match = re.search(
        r"_MUTATIONS:\s*list\[.*?\]\s*=\s*\[(?P<body>.*?)\]\n\n\n#",
        text,
        re.DOTALL,
    )
    if match is None:
        return 0
    return len(re.findall(r'^\s*\("mut_', match.group("body"), re.MULTILINE))


def _count_loader_cases() -> int:
    text = _FUZZ_LOADER.read_text(encoding="utf-8")
    run_cases = re.findall(r'^\s*RUN\("', text, re.MULTILINE)
    path_cases = re.findall(r'^\s*RUNP\("', text, re.MULTILINE)
    return len(run_cases) + len(path_cases)


def _hostile_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(_HOSTILE.glob("*.json")):
        schema = _schema_for_fixture(path.name)
        counts[schema] = counts.get(schema, 0) + 1
    return counts


def _report() -> dict:
    hostile_by_schema = _hostile_counts()
    hostile_total = sum(hostile_by_schema.values())
    json_mutations = _count_json_mutations()
    loader_cases = _count_loader_cases()
    json_total = hostile_total + 2 + json_mutations
    total = loader_cases + json_total

    checks = {
        "hostile_fixtures": hostile_total >= _MIN_HOSTILE_FIXTURES,
        "json_mutations": json_mutations >= _MIN_JSON_MUTATIONS,
        "loader_cases": loader_cases >= _MIN_LOADER_CASES,
    }

    return {
        "status": "pass" if all(checks.values()) else "fail",
        "loader_cases": loader_cases,
        "hostile_fixtures": hostile_total,
        "hostile_by_schema": hostile_by_schema,
        "json_valid_baselines": 2,
        "json_inline_mutations": json_mutations,
        "json_total_cases": json_total,
        "combined_deterministic_cases": total,
        "minimums": {
            "hostile_fixtures": _MIN_HOSTILE_FIXTURES,
            "json_mutations": _MIN_JSON_MUTATIONS,
            "loader_cases": _MIN_LOADER_CASES,
        },
        "checks": checks,
    }


def main() -> int:
    data = _report()
    print("ATT-1 M152 Fuzz Coverage Summary")
    print(f"  Loader cases              : {data['loader_cases']}")
    print(f"  Hostile JSON fixtures      : {data['hostile_fixtures']}")
    for schema, count in sorted(data["hostile_by_schema"].items()):
        print(f"    {schema:<16}: {count}")
    print(f"  JSON valid baselines       : {data['json_valid_baselines']}")
    print(f"  JSON inline mutations      : {data['json_inline_mutations']}")
    print(f"  JSON total cases           : {data['json_total_cases']}")
    print(f"  Combined deterministic     : {data['combined_deterministic_cases']}")
    print(f"  Status                     : {data['status'].upper()}")

    if "--json" in sys.argv:
        print(json.dumps(data, indent=2, sort_keys=True))

    return 0 if data["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
