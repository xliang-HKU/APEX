#!/usr/bin/env python3
"""Validate FiniteTela results against raw averaged stresses."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


VOIGT_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]


@dataclass
class TaskRecord:
    path: str
    temperature: float
    strain_component: int | None
    strain_value: float
    is_reference: bool
    stress_voigt_gpa: List[float]


def _load_json(path: str):
    with open(path, "r") as fh:
        return json.load(fh)


def _safe_float(value) -> float:
    return float(value)


def _average_stress_voigt_gpa(task_dir: str) -> List[float]:
    stress_file = os.path.join(task_dir, "average_stress.txt")
    sums = {
        "xx": 0.0,
        "yy": 0.0,
        "zz": 0.0,
        "xy": 0.0,
        "xz": 0.0,
        "yz": 0.0,
    }
    count = 0

    with open(stress_file, "r") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            _, pxx, pyy, pzz, pxy, pxz, pyz = map(float, parts)
            sums["xx"] += pxx
            sums["yy"] += pyy
            sums["zz"] += pzz
            sums["xy"] += pxy
            sums["xz"] += pxz
            sums["yz"] += pyz
            count += 1

    if count == 0:
        raise RuntimeError(f"No averaged stress data found in {stress_file}")

    # LAMMPS pressure is reported in bar in metal units. FiniteTela converts it
    # to stress by negating and reports final elastic constants in GPa.
    return [
        -sums["xx"] / count / 1e4,
        -sums["yy"] / count / 1e4,
        -sums["zz"] / count / 1e4,
        -sums["yz"] / count / 1e4,
        -sums["xz"] / count / 1e4,
        -sums["xy"] / count / 1e4,
    ]


def _collect_records(workdir: str) -> List[TaskRecord]:
    task_dirs = sorted(glob.glob(os.path.join(workdir, "task.[0-9]*[0-9]")))
    if not task_dirs:
        raise RuntimeError(f"No tasks found under {workdir}")

    records = []
    for task_dir in task_dirs:
        meta_path = os.path.join(task_dir, "FiniteTela.json")
        if not os.path.isfile(meta_path):
            continue
        meta = _load_json(meta_path)
        records.append(
            TaskRecord(
                path=task_dir,
                temperature=_safe_float(meta["temperature"]),
                strain_component=meta["strain_component"],
                strain_value=_safe_float(meta["strain_value"]),
                is_reference=bool(meta["is_reference"]),
                stress_voigt_gpa=_average_stress_voigt_gpa(task_dir),
            )
        )

    if not records:
        raise RuntimeError(f"No FiniteTela metadata found under {workdir}")
    return records


def _group_by_temperature(records: Iterable[TaskRecord]) -> Dict[float, List[TaskRecord]]:
    grouped: Dict[float, List[TaskRecord]] = {}
    for record in records:
        grouped.setdefault(record.temperature, []).append(record)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _find_reference(records: List[TaskRecord]) -> TaskRecord:
    refs = [record for record in records if record.is_reference]
    if len(refs) != 1:
        raise RuntimeError(
            f"Expected exactly one reference task, found {len(refs)}"
        )
    return refs[0]


def _find_pair(records: List[TaskRecord], component: int) -> Tuple[TaskRecord, TaskRecord]:
    pair = [record for record in records if record.strain_component == component]
    if len(pair) != 2:
        raise RuntimeError(
            f"Expected exactly two tasks for strain component {VOIGT_LABELS[component]}, found {len(pair)}"
        )

    pos = [record for record in pair if record.strain_value > 0]
    neg = [record for record in pair if record.strain_value < 0]
    if len(pos) != 1 or len(neg) != 1:
        raise RuntimeError(
            f"Need one positive and one negative strain for {VOIGT_LABELS[component]}"
        )
    return pos[0], neg[0]


def _vector_stats(values: Iterable[float]) -> Tuple[float, float]:
    values = [abs(value) for value in values if not math.isnan(value)]
    if not values:
        return 0.0, 0.0
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    return max(values), rms


def _matrix_diffs(a: List[List[float]], b: List[List[float]]) -> List[float]:
    diffs = []
    for ii in range(len(a)):
        for jj in range(len(a[ii])):
            if math.isnan(a[ii][jj]) or math.isnan(b[ii][jj]):
                continue
            diffs.append(a[ii][jj] - b[ii][jj])
    return diffs


def _matrix_rel_diff(
    a: List[List[float]], b: List[List[float]], floor: float = 1e-8
) -> Tuple[float, float]:
    rel_diffs = []
    for ii in range(len(a)):
        for jj in range(len(a[ii])):
            if math.isnan(a[ii][jj]) or math.isnan(b[ii][jj]):
                continue
            denom = max(abs(b[ii][jj]), floor)
            rel_diffs.append(abs(a[ii][jj] - b[ii][jj]) / denom)
    if not rel_diffs:
        return 0.0, 0.0
    return max(rel_diffs), math.sqrt(sum(x * x for x in rel_diffs) / len(rel_diffs))


def _symmetry_mismatch(matrix: List[List[float]]) -> float:
    diffs = []
    for ii in range(len(matrix)):
        for jj in range(ii + 1, len(matrix[ii])):
            diffs.append(abs(matrix[ii][jj] - matrix[jj][ii]))
    return max(diffs) if diffs else 0.0


def _format_vector(values: List[float], width: int = 9, precision: int = 3) -> str:
    return " ".join(f"{value:{width}.{precision}f}" for value in values)


def _format_matrix(matrix: List[List[float]]) -> str:
    return "\n".join(_format_vector(row) for row in matrix)


def _result_map(workdir: str) -> Dict[float, Dict]:
    result_path = os.path.join(workdir, "result.json")
    if not os.path.isfile(result_path):
        return {}

    result = _load_json(result_path)
    mapped = {}
    for key, value in result.items():
        mapped[float(key)] = value
    return mapped


def _build_manual_tensor(records: List[TaskRecord]) -> Tuple[List[List[float]], Dict[int, Dict], TaskRecord]:
    reference = _find_reference(records)
    manual_tensor = [[float("nan")] * 6 for _ in range(6)]
    component_details = {}

    components = sorted(
        {
            record.strain_component
            for record in records
            if record.strain_component is not None
        }
    )
    for component in components:
        pos, neg = _find_pair(records, component)
        strain = abs(pos.strain_value)
        if not math.isclose(strain, abs(neg.strain_value), rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError(
                f"Positive and negative strain magnitudes differ for {VOIGT_LABELS[component]}"
            )
        if strain <= 0:
            raise RuntimeError(f"Non-positive strain magnitude for {VOIGT_LABELS[component]}")

        column = []
        column_plus = []
        column_minus = []
        for idx in range(6):
            # Rebuild one tensor column from the raw +/- stress response, and
            # keep the one-sided estimates to diagnose finite-temperature noise.
            central = (pos.stress_voigt_gpa[idx] - neg.stress_voigt_gpa[idx]) / (2.0 * strain)
            plus = (pos.stress_voigt_gpa[idx] - reference.stress_voigt_gpa[idx]) / strain
            minus = (reference.stress_voigt_gpa[idx] - neg.stress_voigt_gpa[idx]) / strain
            manual_tensor[idx][component] = central
            column.append(central)
            column_plus.append(plus)
            column_minus.append(minus)

        component_details[component] = {
            "strain": strain,
            "central_column": column,
            "plus_column": column_plus,
            "minus_column": column_minus,
            "linearity_max_abs_gpa": max(
                abs(p - m) for p, m in zip(column_plus, column_minus)
            ),
        }

    return manual_tensor, component_details, reference


def _check_threshold(value: float, abs_tol: float, rel_value: float | None = None, rel_tol: float | None = None) -> bool:
    if value <= abs_tol:
        return False
    if rel_value is None or rel_tol is None:
        return True
    return rel_value > rel_tol


def validate_workdir(args) -> int:
    workdir = os.path.abspath(args.workdir)
    records = _collect_records(workdir)
    grouped = _group_by_temperature(records)
    result_map = _result_map(workdir)

    warnings = 0
    errors = 0
    summary = []
    matched_temperatures = 0

    print(f"FiniteTela validation for: {workdir}")
    print(f"Temperatures found: {', '.join(f'{temp:g}' for temp in grouped)}")
    print()

    for temperature, temp_records in grouped.items():
        if args.temperature is not None and not math.isclose(
            temperature, args.temperature, rel_tol=0.0, abs_tol=1e-8
        ):
            continue
        matched_temperatures += 1
        temp_warnings = 0

        try:
            manual_tensor, details, reference = _build_manual_tensor(temp_records)
        except RuntimeError as exc:
            print(f"[ERROR] T={temperature:g} K: {exc}")
            errors += 1
            continue

        ref_max, ref_rms = _vector_stats(reference.stress_voigt_gpa)
        result_entry = result_map.get(temperature)

        print(f"Temperature {temperature:g} K")
        print(f"  Reference task: {os.path.basename(reference.path)}")
        print(f"  Reference stress max/rms: {ref_max:.4f} / {ref_rms:.4f} GPa")

        if _check_threshold(ref_max, args.eq_stress_tol):
            print(
                f"  [WARN] Residual equilibrium stress exceeds tolerance ({args.eq_stress_tol:.3f} GPa)"
            )
            warnings += 1
            temp_warnings += 1

        if result_entry is None:
            print("  [WARN] result.json missing this temperature, raw-task checks only")
            warnings += 1
            temp_warnings += 1
            print()
            continue

        # Compare the post-processed tensor against a direct finite-difference
        # reconstruction from average_stress.txt.
        fit_tensor = result_entry["elastic_tensor"]
        diff_values = _matrix_diffs(fit_tensor, manual_tensor)
        diff_max, diff_rms = _vector_stats(diff_values)
        rel_max, rel_rms = _matrix_rel_diff(fit_tensor, manual_tensor)
        sym_max = _symmetry_mismatch(fit_tensor)

        print(f"  Fit-vs-central diff max/rms: {diff_max:.3f} / {diff_rms:.3f} GPa")
        print(f"  Fit-vs-central rel max/rms: {rel_max:.3%} / {rel_rms:.3%}")
        print(f"  Tensor symmetry max |Cij-Cji|: {sym_max:.3f} GPa")
        print(
            f"  Moduli from result: B={result_entry['B']:.3f} GPa, G={result_entry['G']:.3f} GPa, "
            f"E={result_entry['E']:.3f} GPa, u={result_entry['u']:.4f}"
        )

        if _check_threshold(diff_max, args.fit_abs_tol, rel_max, args.fit_rel_tol):
            print(
                f"  [WARN] Fitted tensor differs from central difference beyond tolerances "
                f"({args.fit_abs_tol:.3f} GPa, {args.fit_rel_tol:.1%})"
            )
            warnings += 1
            temp_warnings += 1

        if _check_threshold(sym_max, args.sym_tol):
            print(
                f"  [WARN] Tensor symmetry mismatch exceeds tolerance ({args.sym_tol:.3f} GPa)"
            )
            warnings += 1
            temp_warnings += 1

        if result_entry["B"] <= 0 or result_entry["G"] <= 0 or result_entry["E"] <= 0:
            print("  [WARN] One or more elastic moduli are non-positive")
            warnings += 1
            temp_warnings += 1

        if args.verbose:
            print("  Fitted tensor (GPa):")
            print(_indent_block(_format_matrix(fit_tensor), "    "))
            print("  Central-difference tensor (GPa):")
            print(_indent_block(_format_matrix(manual_tensor), "    "))
            for component in sorted(details):
                detail = details[component]
                rel_linearity = _relative_column_gap(
                    detail["plus_column"], detail["minus_column"]
                )
                print(
                    f"  d/d{VOIGT_LABELS[component]}: strain={detail['strain']:.6f}, "
                    f"linearity max={detail['linearity_max_abs_gpa']:.3f} GPa, "
                    f"rel={rel_linearity:.3%}"
                )
                print(
                    _indent_block(
                        "central " + _format_vector(detail["central_column"]),
                        "    ",
                    )
                )
                print(
                    _indent_block(
                        "plus    " + _format_vector(detail["plus_column"]),
                        "    ",
                    )
                )
                print(
                    _indent_block(
                        "minus   " + _format_vector(detail["minus_column"]),
                        "    ",
                    )
                )

        for component, detail in details.items():
            rel_linearity = _relative_column_gap(
                detail["plus_column"], detail["minus_column"]
            )
            if _check_threshold(
                detail["linearity_max_abs_gpa"],
                args.linearity_abs_tol,
                rel_linearity,
                args.linearity_rel_tol,
            ):
                print(
                    f"  [WARN] d/d{VOIGT_LABELS[component]} shows strong +/- asymmetry "
                    f"({detail['linearity_max_abs_gpa']:.3f} GPa, {rel_linearity:.3%})"
                )
                warnings += 1
                temp_warnings += 1

        summary.append(
            {
                "temperature": temperature,
                "reference_stress_max_gpa": ref_max,
                "fit_central_max_diff_gpa": diff_max,
                "fit_central_rms_diff_gpa": diff_rms,
                "fit_central_max_rel": rel_max,
                "tensor_symmetry_max_gpa": sym_max,
                "warning_count": temp_warnings,
            }
        )
        print()

    if args.temperature is not None and matched_temperatures == 0:
        print(f"[ERROR] Requested temperature {args.temperature:g} K was not found.")
        return 2

    if args.summary_json:
        with open(args.summary_json, "w") as fh:
            json.dump(summary, fh, indent=2)

    if errors:
        print(f"Validation finished with {errors} error(s) and {warnings} warning(s).")
        return 2
    if warnings and args.strict:
        print(f"Validation finished with {warnings} warning(s); strict mode treats this as failure.")
        return 1

    print(f"Validation finished with {warnings} warning(s).")
    return 0


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _relative_column_gap(a: List[float], b: List[float], floor: float = 1e-8) -> float:
    ratios = []
    for va, vb in zip(a, b):
        # Normalize by the larger one-sided response so near-zero components do
        # not dominate the relative asymmetry metric.
        denom = max(max(abs(va), abs(vb)), floor)
        ratios.append(abs(va - vb) / denom)
    return max(ratios) if ratios else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a FiniteTela work directory by comparing fitted elastic tensors "
            "against raw central differences from average_stress.txt."
        )
    )
    parser.add_argument("workdir", help="Path to a finitetela_* work directory")
    parser.add_argument(
        "--temperature",
        type=float,
        help="Validate only one temperature (exact float match)",
    )
    parser.add_argument(
        "--eq-stress-tol",
        type=float,
        default=1.0,
        help="Warning threshold for residual equilibrium stress in GPa (default: 1.0)",
    )
    parser.add_argument(
        "--fit-abs-tol",
        type=float,
        default=5.0,
        help="Warning threshold for fit-vs-central max absolute difference in GPa (default: 5.0)",
    )
    parser.add_argument(
        "--fit-rel-tol",
        type=float,
        default=0.10,
        help="Warning threshold for fit-vs-central max relative difference (default: 0.10)",
    )
    parser.add_argument(
        "--linearity-abs-tol",
        type=float,
        default=10.0,
        help="Warning threshold for +/- one-sided asymmetry in GPa (default: 10.0)",
    )
    parser.add_argument(
        "--linearity-rel-tol",
        type=float,
        default=0.15,
        help="Warning threshold for +/- one-sided asymmetry in relative terms (default: 0.15)",
    )
    parser.add_argument(
        "--sym-tol",
        type=float,
        default=5.0,
        help="Warning threshold for max |Cij-Cji| in GPa (default: 5.0)",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path to write a JSON summary",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tensors and per-component central/one-sided columns",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 if any warning is triggered",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return validate_workdir(args)


if __name__ == "__main__":
    raise SystemExit(main())
