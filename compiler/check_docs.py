#!/usr/bin/env python3
"""
ATT-1 M149: documentation lint and link checker.

Validates internal Markdown links, required docs, forbidden patterns,
and basic milestone/status consistency for the ATT-1 repository.

Checks performed:
  1. Internal Markdown links — target file exists; warns on missing anchors.
  2. Required docs — all 18 key documents from docs/INDEX.md and README.md
     must be present on disk.
  3. Forbidden/stale patterns — tracked __pycache__/.pyc/.pyo files,
     absolute local paths in docs, stale "future manual" claims.
  4. Milestone/status consistency — OPERATION_LOG mentions M157;
     M156 marked complete; M146/M147 manuals not still "pending".

This tool does NOT access the network, execute inference, run real MMIO,
require CUDA, or modify any file.  All checking is static analysis only.

Exit codes:
  0 — all checks pass (warnings may be present)
  1 — one or more lint/link failures detected
  2 — tool/parser error (unexpected exception)

Usage:
    python3 compiler/check_docs.py
    python3 compiler/check_docs.py --report-json report.json
    python3 compiler/check_docs.py --warn-anchors
    python3 compiler/check_docs.py --repo-root /path/to/att1

    make docs-check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent   # compiler/
_DEFAULT_REPO = _HERE.parent              # project root


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    level: str        # "error" | "warning"
    category: str
    file: str
    message: str
    line: int = 0


@dataclass
class CheckReport:
    issues: list[Issue] = field(default_factory=list)
    checks_run: int = 0
    files_scanned: int = 0

    def add_error(self, category: str, file: str, message: str, line: int = 0) -> None:
        self.issues.append(Issue("error", category, file, message, line))

    def add_warning(self, category: str, file: str, message: str, line: int = 0) -> None:
        self.issues.append(Issue("warning", category, file, message, line))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]


# ---------------------------------------------------------------------------
# Required documents
# ---------------------------------------------------------------------------

_REQUIRED_DOCS: list[str] = [
    "README.md",
    "docs/INDEX.md",
    "docs/ATT1_REFERENCE_MANUAL.md",
    "docs/AIMU_INTRINSICS_OPERATIONS_REFERENCE.md",
    "docs/RELEASE_READINESS.md",
    "docs/RELEASE_CANDIDATE_M150.md",
    "docs/CUDA_SIGNOFF_M155.md",
    "docs/EXTERNAL_REVIEW_PACKAGE.md",
    "docs/testing.md",
    "docs/CUDA_VALIDATION_PLAN.md",
    "docs/OPERATION_LOG.md",
    "docs/aimu_architecture.md",
    "docs/aimu_pcie_prototype_review.md",
    "docs/aimu_pcie_command_requirements.md",
    "docs/aimu_register_map.md",
    "docs/aimu_fabric_routing.md",
    "docs/tensor_placement_report.md",
    "docs/tensor_execution_plan.md",
    "docs/PHASE1_TO_PHASE2_GAP_AUDIT.md",
]


# ---------------------------------------------------------------------------
# Patterns for absolute local paths that should not appear in docs
# ---------------------------------------------------------------------------

# Allow these patterns in historical OPERATION_LOG entries and tool outputs
_ABS_PATH_PATTERN = re.compile(
    r"(?<![`'\"])(/home/[a-zA-Z0-9_.-]+|/usr/export/[a-zA-Z0-9_.-]+)"
)

# Filenames/dirs that should never appear as git-tracked artifacts
_FORBIDDEN_TRACKED: list[str] = ["__pycache__", ".pyc", ".pyo"]


# ---------------------------------------------------------------------------
# Markdown link extraction
# ---------------------------------------------------------------------------

# Matches [text](target) but not ![alt](img) differently — we parse both
_MD_LINK_RE = re.compile(r"\[(?:[^\]]*)\]\(([^)]+)\)")


_CODE_SPAN_RE = re.compile(r"`[^`]*`")


def _parse_md_links(text: str) -> list[tuple[int, str]]:
    """Return list of (line_number, target) for all Markdown links in text."""
    results: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        # Strip inline code spans so links inside backticks are not resolved
        clean = _CODE_SPAN_RE.sub(lambda m: " " * len(m.group()), line)
        for m in _MD_LINK_RE.finditer(clean):
            target = m.group(1).strip()
            results.append((lineno, target))
    return results


def _is_external_link(target: str) -> bool:
    return target.startswith(("http://", "https://", "ftp://", "mailto:"))


def _split_anchor(target: str) -> tuple[str, Optional[str]]:
    """Split 'file.md#anchor' into ('file.md', 'anchor')."""
    if "#" in target:
        idx = target.index("#")
        return target[:idx], target[idx + 1:] or None
    return target, None


# ---------------------------------------------------------------------------
# Check 1: internal markdown link targets exist
# ---------------------------------------------------------------------------

def check_links(repo: Path, report: CheckReport, warn_anchors: bool) -> None:
    """For every .md file, verify internal link targets exist on disk."""
    md_files = sorted(repo.rglob("*.md"))
    # Exclude build dirs and hidden dirs
    md_files = [
        f for f in md_files
        if not any(part.startswith(".") or part in ("build", "build-asan", "build-ubsan")
                   for part in f.parts)
    ]
    report.files_scanned += len(md_files)
    report.checks_run += 1

    for md_path in md_files:
        rel_source = str(md_path.relative_to(repo))
        try:
            text = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            report.add_error("links", rel_source, f"Cannot read file: {exc}")
            continue

        links = _parse_md_links(text)
        for lineno, raw_target in links:
            if _is_external_link(raw_target):
                continue

            # Strip leading whitespace/newlines inside target (shouldn't
            # happen but be defensive)
            raw_target = raw_target.strip()
            if not raw_target:
                continue

            file_part, anchor = _split_anchor(raw_target)
            if not file_part:
                # Link is anchor-only (same-page reference) — skip
                continue

            # Resolve relative to the source file's directory
            target_path = (md_path.parent / file_part).resolve()

            if not target_path.exists():
                report.add_error(
                    "links",
                    rel_source,
                    f"Line {lineno}: broken link → '{raw_target}' (resolved: {target_path})",
                    line=lineno,
                )
            elif anchor and warn_anchors:
                # Best-effort anchor validation: look for the heading in target
                try:
                    target_text = target_path.read_text(encoding="utf-8", errors="replace")
                    # GitHub-style anchor: lowercase, spaces→hyphens, strip non-alnum
                    def _heading_anchor(heading: str) -> str:
                        s = heading.lower()
                        s = re.sub(r"[^\w\s-]", "", s)
                        s = re.sub(r"\s+", "-", s.strip())
                        return s

                    headings_in_target = re.findall(r"^#{1,6}\s+(.*)", target_text, re.MULTILINE)
                    anchors_in_target = {_heading_anchor(h) for h in headings_in_target}
                    if anchor not in anchors_in_target:
                        report.add_warning(
                            "anchors",
                            rel_source,
                            f"Line {lineno}: anchor '#{anchor}' not found in '{file_part}'",
                            line=lineno,
                        )
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Check 2: required documents exist
# ---------------------------------------------------------------------------

def check_required_docs(repo: Path, report: CheckReport) -> None:
    report.checks_run += 1
    for rel_path in _REQUIRED_DOCS:
        full = repo / rel_path
        if not full.exists():
            report.add_error(
                "required-docs",
                rel_path,
                f"Required document is missing: {rel_path}",
            )


# ---------------------------------------------------------------------------
# Check 3a: tracked Python cache artifacts
# ---------------------------------------------------------------------------

def check_tracked_cache_artifacts(repo: Path, report: CheckReport) -> None:
    report.checks_run += 1
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            report.add_warning(
                "hygiene",
                "(git)",
                f"git ls-files failed (rc={proc.returncode}); skipping cache artifact check",
            )
            return
        for line in proc.stdout.splitlines():
            if "__pycache__" in line or line.endswith(".pyc") or line.endswith(".pyo"):
                report.add_error(
                    "hygiene",
                    line,
                    f"Tracked Python cache artifact: {line}",
                )
    except OSError as exc:
        report.add_warning("hygiene", "(git)", f"Cannot run git ls-files: {exc}")


# ---------------------------------------------------------------------------
# Check 3b: absolute local paths in docs
# ---------------------------------------------------------------------------

def check_absolute_paths(repo: Path, report: CheckReport) -> None:
    """
    Warn when absolute local paths like /home/rick appear in doc files.

    The OPERATION_LOG contains historical tool output that may include
    absolute paths (e.g. from run_full_regression.py --report-json output).
    We warn rather than error to avoid false positives on historical entries.
    """
    report.checks_run += 1
    doc_files = list((repo / "docs").glob("*.md")) + [repo / "README.md"]
    for doc_path in doc_files:
        if not doc_path.exists():
            continue
        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel_source = str(doc_path.relative_to(repo))
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in _ABS_PATH_PATTERN.finditer(line):
                abs_path = m.group(1)
                report.add_warning(
                    "absolute-paths",
                    rel_source,
                    f"Line {lineno}: absolute local path '{abs_path}' in doc",
                    line=lineno,
                )


# ---------------------------------------------------------------------------
# Check 3c: stale "future manual" claims
# ---------------------------------------------------------------------------

_STALE_FUTURE_MANUAL_PATTERNS: list[tuple[str, str]] = [
    # Pattern, description
    (
        r"planned for future milestones",
        "stale 'planned for future milestones' claim (M146/M147 are complete)",
    ),
    (
        r"Future Reference Manuals",
        "stale section title 'Future Reference Manuals' (both manuals now complete)",
    ),
]


def check_stale_claims(repo: Path, report: CheckReport) -> None:
    report.checks_run += 1
    doc_files = sorted((repo / "docs").glob("*.md")) + [repo / "README.md"]
    # OPERATION_LOG is a historical write-once log; its milestone entries
    # legitimately describe what old sections were called. Skip it here.
    _STALE_SKIP = {"OPERATION_LOG.md"}
    for doc_path in doc_files:
        if doc_path.name in _STALE_SKIP:
            continue
        if not doc_path.exists():
            continue
        rel_source = str(doc_path.relative_to(repo))
        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern, desc in _STALE_FUTURE_MANUAL_PATTERNS:
            for lineno, line in enumerate(text.splitlines(), start=1):
                # Skip matches inside inline code spans
                clean = _CODE_SPAN_RE.sub(lambda m: " " * len(m.group()), line)
                if re.search(pattern, clean, re.IGNORECASE):
                    report.add_error(
                        "stale-claims",
                        rel_source,
                        f"Line {lineno}: {desc}",
                        line=lineno,
                    )


# ---------------------------------------------------------------------------
# Check 4: milestone/status consistency
# ---------------------------------------------------------------------------

def check_milestone_consistency(repo: Path, report: CheckReport) -> None:
    """
    Validate key milestone/status invariants:
      - OPERATION_LOG must contain a Milestone 158 entry.
      - OPERATION_LOG must show M157 as complete.
      - CPU CI must be described as CPU-only somewhere in testing docs.
      - CUDA signoff described as manual RTX 3090.
    """
    report.checks_run += 1

    op_log = repo / "docs" / "OPERATION_LOG.md"
    if op_log.exists():
        text = op_log.read_text(encoding="utf-8", errors="replace")

        # M158 should appear (at minimum in "Next Prompt for Codex" or as an entry)
        if not re.search(r"Milestone\s+158|M158", text):
            report.add_error(
                "milestone-consistency",
                "docs/OPERATION_LOG.md",
                "OPERATION_LOG does not mention Milestone 158 (M158 not started or missing)",
            )

        # M157 should be listed as complete (have a "Milestone 157:" entry)
        if not re.search(r"^- Milestone 157:", text, re.MULTILINE):
            report.add_error(
                "milestone-consistency",
                "docs/OPERATION_LOG.md",
                "OPERATION_LOG has no '- Milestone 157:' entry (M157 not yet recorded as complete)",
            )

    # CPU CI described as CPU-only in testing.md or RELEASE_READINESS
    testing_md = repo / "docs" / "testing.md"
    if testing_md.exists():
        text = testing_md.read_text(encoding="utf-8", errors="replace")
        if "cpu-only" not in text.lower() and "cpu only" not in text.lower():
            report.add_warning(
                "milestone-consistency",
                "docs/testing.md",
                "testing.md does not mention CPU-only CI policy",
            )

    # CUDA signoff described as manual RTX 3090
    cuda_plan = repo / "docs" / "CUDA_VALIDATION_PLAN.md"
    if cuda_plan.exists():
        text = cuda_plan.read_text(encoding="utf-8", errors="replace")
        if "rtx 3090" not in text.lower():
            report.add_warning(
                "milestone-consistency",
                "docs/CUDA_VALIDATION_PLAN.md",
                "CUDA_VALIDATION_PLAN.md does not mention RTX 3090 (expected for manual signoff)",
            )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_report(report: CheckReport, verbose: bool = False) -> None:
    print()
    print("=" * 72)
    print("ATT-1 M149 Documentation Lint and Link Checker")
    print(f"  Files scanned : {report.files_scanned}")
    print(f"  Checks run    : {report.checks_run}")
    print(f"  Errors        : {len(report.errors)}")
    print(f"  Warnings      : {len(report.warnings)}")
    print("=" * 72)

    if report.errors:
        print()
        print("ERRORS:")
        for issue in report.errors:
            loc = f"{issue.file}" + (f":{issue.line}" if issue.line else "")
            print(f"  [ERROR/{issue.category}] {loc}")
            print(f"    {issue.message}")

    if report.warnings:
        print()
        print("WARNINGS:")
        for issue in report.warnings:
            loc = f"{issue.file}" + (f":{issue.line}" if issue.line else "")
            print(f"  [WARN/{issue.category}] {loc}")
            print(f"    {issue.message}")

    print()
    if not report.errors:
        print("Result: PASS")
    else:
        print(f"Result: FAIL ({len(report.errors)} error(s))")
    print()


def _write_json_report(report: CheckReport, path: str) -> None:
    data = {
        "tool": "check_docs",
        "milestone": "M149",
        "files_scanned": report.files_scanned,
        "checks_run": report.checks_run,
        "overall": "pass" if not report.errors else "fail",
        "error_count": len(report.errors),
        "warning_count": len(report.warnings),
        "issues": [
            {
                "level": i.level,
                "category": i.category,
                "file": i.file,
                "line": i.line,
                "message": i.message,
            }
            for i in report.issues
        ],
    }
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        print(f"JSON report written to: {path}")
    except OSError as exc:
        print(f"warning: could not write JSON report: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ATT-1 M149: documentation lint and link checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--repo-root",
        metavar="DIR",
        dest="repo_root",
        default=None,
        help="Path to the ATT-1 repository root (default: parent of this script).",
    )
    ap.add_argument(
        "--report-json",
        metavar="FILE",
        dest="report_json",
        help="Write a JSON report to this file.",
    )
    ap.add_argument(
        "--warn-anchors",
        action="store_true",
        dest="warn_anchors",
        help="Warn when Markdown link anchors cannot be verified in the target file.",
    )
    ap.add_argument(
        "--no-path-warnings",
        action="store_true",
        dest="no_path_warnings",
        help="Suppress warnings about absolute local paths in docs.",
    )
    return ap.parse_args()


def main() -> int:
    try:
        args = _parse_args()
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0

    repo = Path(args.repo_root).resolve() if args.repo_root else _DEFAULT_REPO

    if not repo.is_dir():
        print(f"error: repo root not found: {repo}", file=sys.stderr)
        return 2

    report = CheckReport()

    print()
    print("ATT-1 M149 Documentation Lint and Link Checker")
    print(f"  Repo: {repo}")
    print()

    try:
        print("[check] Required documents exist ...")
        check_required_docs(repo, report)

        print("[check] Internal Markdown links ...")
        check_links(repo, report, warn_anchors=args.warn_anchors)

        print("[check] Tracked Python cache artifacts ...")
        check_tracked_cache_artifacts(repo, report)

        if not args.no_path_warnings:
            print("[check] Absolute local paths in docs ...")
            check_absolute_paths(repo, report)

        print("[check] Stale 'future manual' claims ...")
        check_stale_claims(repo, report)

        print("[check] Milestone/status consistency ...")
        check_milestone_consistency(repo, report)

    except Exception as exc:  # noqa: BLE001
        print(f"error: unexpected exception: {exc}", file=sys.stderr)
        return 2

    _print_report(report)

    if args.report_json:
        _write_json_report(report, args.report_json)

    return 0 if not report.errors else 1


if __name__ == "__main__":
    sys.exit(main())
