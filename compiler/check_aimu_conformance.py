#!/usr/bin/env python3
"""
ATT-1 M161: static conformance checker for the frozen AIMU interfaces.

Checks that the frozen v1.0 documentation and C headers remain aligned for:
  - register-map version and selected MMIO offsets
  - command packet size / command-type values / result-code values
  - DMA descriptor size / enum values / flag bits / frozen field names
  - fabric counter-name set frozen at M160

This tool is documentation/header consistency only.  It does not run the C
simulator, require CUDA, or access hardware.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


@dataclass
class Issue:
    level: str
    category: str
    file: str
    message: str


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    checks_run: int = 0

    def add_error(self, category: str, file: str, message: str) -> None:
        self.issues.append(Issue("error", category, file, message))

    def add_warning(self, category: str, file: str, message: str) -> None:
        self.issues.append(Issue("warning", category, file, message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]


def _parse_macro_values(header_text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    pattern = re.compile(r"^#define\s+(\w+)\s+(.+?)\s*(?:/\*.*)?$", re.MULTILINE)
    for name, raw_expr in pattern.findall(header_text):
        expr = raw_expr.split("/*", 1)[0].strip()
        if name.endswith("(N)"):
            continue
        expr = expr.replace("UINT32_C", "")
        expr = expr.replace("UINT64_C", "")
        expr = expr.replace("UINT8_C", "")
        expr = expr.replace("UINT16_C", "")
        expr = expr.replace("(", "").replace(")", "")
        expr = expr.strip()
        if not expr:
            continue
        if re.fullmatch(r"0x[0-9A-Fa-f]+", expr):
            values[name] = int(expr, 16)
        elif re.fullmatch(r"[0-9]+u?", expr):
            values[name] = int(expr.rstrip("u"), 10)
    return values


def _parse_enum_values(header_text: str, enum_name: str) -> dict[str, int]:
    match = re.search(
        rf"typedef\s+enum\s+{re.escape(enum_name)}\s*\{{(.*?)\}}\s*{re.escape(enum_name)}\s*;",
        header_text,
        re.DOTALL,
    )
    if not match:
        return {}
    body = match.group(1)
    values: dict[str, int] = {}
    for raw_line in body.splitlines():
        line = raw_line.split("/*", 1)[0].split("//", 1)[0].strip().rstrip(",")
        if not line or "=" not in line:
            continue
        name, raw_value = [part.strip() for part in line.split("=", 1)]
        raw_value = raw_value.replace("UINT32_C", "").replace("UINT8_C", "")
        raw_value = raw_value.replace("(", "").replace(")", "").strip()
        base = 16 if raw_value.lower().startswith("0x") else 10
        values[name] = int(raw_value.rstrip("u"), base)
    return values


def _parse_struct_fields(header_text: str, struct_name: str) -> list[str]:
    match = re.search(
        rf"typedef\s+struct\s+{re.escape(struct_name)}\s*\{{(.*?)\}}\s*{re.escape(struct_name)}\s*;",
        header_text,
        re.DOTALL,
    )
    if not match:
        return []
    fields: list[str] = []
    body = match.group(1)
    for raw_line in body.splitlines():
        line = raw_line.split("/*", 1)[0].split("//", 1)[0].strip()
        if not line or line.startswith("*"):
            continue
        if ";" not in line:
            continue
        field = line.rstrip(";").split()[-1]
        field = field.split("[", 1)[0].lstrip("*")
        fields.append(field)
    return fields


def _require_text(report: Report, category: str, path: Path, text: str, pattern: str, message: str) -> None:
    if re.search(pattern, text, re.MULTILINE | re.DOTALL) is None:
        report.add_error(category, str(path.relative_to(_REPO)), message)


def check_register_map(repo: Path, report: Report) -> None:
    report.checks_run += 1
    header_path = repo / "include" / "att1_aimu_mmio.h"
    device_header_path = repo / "include" / "att1_aimu_device.h"
    doc_path = repo / "docs" / "aimu_register_map.md"
    header_text = _read(header_path)
    device_text = _read(device_header_path)
    doc_text = _read(doc_path)
    macros = _parse_macro_values(header_text)
    device_macros = _parse_macro_values(device_text)

    expected = {
        "ATT1_AIMU_REGISTER_MAP_VERSION": 0x00010000,
        "ATT1_MMIO_DEVICE_ID": 0x0000,
        "ATT1_MMIO_REGISTER_MAP_VERSION": 0x0008,
        "ATT1_MMIO_COMMAND_QUEUE_COUNT": 0x0018,
        "ATT1_MMIO_CQ_DOORBELL": 0x1014,
        "ATT1_MMIO_DMA_CONTROL": 0x2000,
        "ATT1_MMIO_FABRIC_STATUS": 0x3000,
        "ATT1_MMIO_CNT_CMD_ISSUED_LO": 0x4000,
        "ATT1_MMIO_TRACE_SNAPSHOT_CONTROL": 0x5018,
    }
    merged = dict(macros)
    merged.update(device_macros)
    for name, value in expected.items():
        if merged.get(name) != value:
            report.add_error(
                "register-map",
                str(header_path.relative_to(repo)),
                f"{name} expected 0x{value:08X}, found {merged.get(name)!r}",
            )

    _require_text(report, "register-map", doc_path, doc_text,
                  r"REGISTER_MAP_VERSION.*0x0001_0000",
                  "aimu_register_map.md does not declare REGISTER_MAP_VERSION = 0x0001_0000")
    _require_text(report, "register-map", doc_path, doc_text,
                  r"### 2\.3 `REGISTER_MAP_VERSION`.*offset 0x0008",
                  "aimu_register_map.md does not pin REGISTER_MAP_VERSION to offset 0x0008")
    _require_text(report, "register-map", doc_path, doc_text,
                  r"### 2\.7 `COMMAND_QUEUE_COUNT`.*offset 0x0018",
                  "aimu_register_map.md does not pin COMMAND_QUEUE_COUNT to offset 0x0018")
    _require_text(report, "register-map", doc_path, doc_text,
                  r"### 5\.3 `DMA_CONTROL`.*offset 0x2000",
                  "aimu_register_map.md does not pin DMA_CONTROL to offset 0x2000")
    _require_text(report, "register-map", doc_path, doc_text,
                  r"### 6\.1 `FABRIC_STATUS`.*offset 0x3000",
                  "aimu_register_map.md does not pin FABRIC_STATUS to offset 0x3000")


def check_command_schema(repo: Path, report: Report) -> None:
    report.checks_run += 1
    header_path = repo / "include" / "att1_aimu_cmdq.h"
    doc_path = repo / "docs" / "aimu_pcie_command_requirements.md"
    header_text = _read(header_path)
    doc_text = _read(doc_path)
    cmd_values = _parse_enum_values(header_text, "att1_aimu_cmd_type")
    result_values = _parse_enum_values(header_text, "att1_aimu_result")

    if "sizeof(att1_aimu_cmd) == 64u" not in header_text:
        report.add_error("command-schema", str(header_path.relative_to(repo)),
                         "att1_aimu_cmd is not guarded as a 64-byte packet")

    expected_cmds = {
        "ATT1_AIMU_CMD_NOP": 0x00,
        "ATT1_AIMU_CMD_LOAD_TENSOR_TILE": 0x01,
        "ATT1_AIMU_CMD_EXEC_MATMUL": 0x10,
        "ATT1_AIMU_CMD_TILE_BARRIER": 0x41,
        "ATT1_AIMU_CMD_QUERY_COUNTERS": 0x51,
    }
    for name, value in expected_cmds.items():
        if cmd_values.get(name) != value:
            report.add_error("command-schema", str(header_path.relative_to(repo)),
                             f"{name} expected 0x{value:02X}, found {cmd_values.get(name)!r}")

    expected_results = {
        "ATT1_AIMU_OK": 0x00,
        "ATT1_AIMU_ERR_QUEUE_FULL": 0x21,
        "ATT1_AIMU_ERR_UNSUPPORTED_OP": 0x60,
        "ATT1_AIMU_ERR_FATAL": 0xFF,
    }
    for name, value in expected_results.items():
        if result_values.get(name) != value:
            report.add_error("command-schema", str(header_path.relative_to(repo)),
                             f"{name} expected 0x{value:02X}, found {result_values.get(name)!r}")

    _require_text(report, "command-schema", doc_path, doc_text,
                  r"64-byte command packet layout",
                  "command requirements doc no longer declares the 64-byte command packet layout")
    _require_text(report, "command-schema", doc_path, doc_text,
                  r"`LOAD_TENSOR_TILE`\s+`0x01`",
                  "command requirements doc no longer freezes LOAD_TENSOR_TILE = 0x01")
    _require_text(report, "command-schema", doc_path, doc_text,
                  r"`QUERY_COUNTERS`\s+`0x51`",
                  "command requirements doc no longer freezes QUERY_COUNTERS = 0x51")
    _require_text(report, "command-schema", doc_path, doc_text,
                  r"`ERR_NONE` `0x00` through\s+`ERR_FATAL` `0xFF`",
                  "command requirements doc no longer freezes the error/result code span")


def check_dma_schema(repo: Path, report: Report) -> None:
    report.checks_run += 1
    header_path = repo / "include" / "att1_aimu_dma.h"
    doc_path = repo / "docs" / "aimu_register_map.md"
    compat_path = repo / "docs" / "schema_compatibility.md"
    header_text = _read(header_path)
    doc_text = _read(doc_path)
    compat_text = _read(compat_path)
    macros = _parse_macro_values(header_text)
    directions = _parse_enum_values(header_text, "att1_aimu_dma_direction")
    fields = _parse_struct_fields(header_text, "att1_aimu_dma_desc")

    if "sizeof(att1_aimu_dma_desc) == 64u" not in header_text:
        report.add_error("dma-schema", str(header_path.relative_to(repo)),
                         "att1_aimu_dma_desc is not guarded as a 64-byte descriptor")

    expected_macros = {
        "ATT1_AIMU_DMA_DTYPE_F32": 0,
        "ATT1_AIMU_DMA_DTYPE_Q8": 1,
        "ATT1_AIMU_DMA_DTYPE_Q4": 2,
        "ATT1_AIMU_DMA_FLAG_VALIDATE_CHECKSUM": 0x0001,
        "ATT1_AIMU_DMA_FLAG_GENERATE_CHECKSUM": 0x0002,
        "ATT1_AIMU_DMA_FLAG_LAST_DESCRIPTOR": 0x0004,
        "ATT1_AIMU_DMA_FLAG_SCATTER_GATHER": 0x0008,
        "ATT1_AIMU_DMA_FLAG_VALID_MASK": 0x000F,
    }
    for name, value in expected_macros.items():
        if macros.get(name) != value:
            report.add_error("dma-schema", str(header_path.relative_to(repo)),
                             f"{name} expected 0x{value:04X}, found {macros.get(name)!r}")

    expected_directions = {
        "ATT1_AIMU_DMA_HOST_TO_DEVICE": 0,
        "ATT1_AIMU_DMA_DEVICE_TO_HOST": 1,
        "ATT1_AIMU_DMA_DEVICE_TO_DEVICE": 2,
    }
    for name, value in expected_directions.items():
        if directions.get(name) != value:
            report.add_error("dma-schema", str(header_path.relative_to(repo)),
                             f"{name} expected {value}, found {directions.get(name)!r}")

    frozen_fields = [
        "host_addr", "device_addr", "src_device_addr", "dst_device_addr",
        "byte_length", "descriptor_id", "command_id", "tensor_id",
        "checksum", "dim0", "dim1", "flags", "dtype",
        "quant_group_size", "direction", "_pad",
    ]
    for field in frozen_fields:
        if field not in fields:
            report.add_error("dma-schema", str(header_path.relative_to(repo)),
                             f"att1_aimu_dma_desc is missing frozen field {field}")
        if field not in doc_text:
            report.add_error("dma-schema", str(doc_path.relative_to(repo)),
                             f"aimu_register_map.md no longer mentions frozen DMA field {field}")

    _require_text(report, "dma-schema", doc_path, doc_text,
                  r"ATT1_AIMU_DMA_FLAG_VALID_MASK = 0x000F",
                  "aimu_register_map.md does not declare ATT1_AIMU_DMA_FLAG_VALID_MASK = 0x000F")
    _require_text(report, "dma-schema", doc_path, doc_text,
                  r"HOST_TO_DEVICE=0`, `DEVICE_TO_HOST=1`, `DEVICE_TO_DEVICE=2",
                  "aimu_register_map.md does not freeze DMA direction values 0/1/2")
    _require_text(report, "dma-schema", compat_path, compat_text,
                  r"must accept the exact v1\.0 `att1_aimu_dma_desc` layout, enum values, flag-bit",
                  "schema_compatibility.md no longer points consumers at the frozen DMA contract")


def check_fabric_counters(repo: Path, report: Report) -> None:
    report.checks_run += 1
    header_path = repo / "include" / "att1_fabric.h"
    doc_path = repo / "docs" / "aimu_fabric_routing.md"
    header_text = _read(header_path)
    doc_text = _read(doc_path)
    fields = _parse_struct_fields(header_text, "att1_fabric_counters")

    for field in fields:
        if field not in doc_text:
            report.add_error("fabric-counters", str(doc_path.relative_to(repo)),
                             f"aimu_fabric_routing.md does not mention frozen fabric counter {field}")

    _require_text(report, "fabric-counters", doc_path, doc_text,
                  r"preflighted and all-or-nothing",
                  "aimu_fabric_routing.md no longer freezes broadcast all-or-nothing semantics")
    _require_text(report, "fabric-counters", doc_path, doc_text,
                  r"The barrier is single-generation and all-or-nothing",
                  "aimu_fabric_routing.md no longer freezes single-generation all-or-nothing barriers")
    _require_text(report, "fabric-counters", doc_path, doc_text,
                  r"queue_full_errors` increments once",
                  "aimu_fabric_routing.md no longer freezes queue_full_errors increment semantics")


def run(repo: Path) -> Report:
    report = Report()
    check_register_map(repo, report)
    check_command_schema(repo, report)
    check_dma_schema(repo, report)
    check_fabric_counters(repo, report)
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check frozen AIMU conformance constants against docs and headers")
    parser.add_argument("--repo-root", default=str(_REPO))
    parser.add_argument("--report-json")
    args = parser.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    report = run(repo)
    overall = "pass" if not report.errors else "fail"

    for issue in report.issues:
        print(f"[{issue.level.upper():7}] {issue.category}: {issue.file}: {issue.message}")

    print(f"Checks run: {report.checks_run}")
    print(f"Errors: {len(report.errors)}  Warnings: {len(report.warnings)}")
    print(f"Result: {overall.upper()}")

    if args.report_json:
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "tool": "check_aimu_conformance.py",
            "overall": overall,
            "checks_run": report.checks_run,
            "error_count": len(report.errors),
            "warning_count": len(report.warnings),
            "issues": [issue.__dict__ for issue in report.issues],
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
