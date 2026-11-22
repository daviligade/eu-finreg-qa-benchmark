"""Turn a finished run into result tables.

Reads runs/<dir>/metrics.csv and answers.jsonl and writes four tables,
each as console text and as a LaTeX fragment under <dir>/paper_tables/.

  T1  headline: the six deterministic metrics per system
  T2  judged metrics (RAGAS) on the COMMON ANSWERED subset only
  T3  refusal calibration: refusal rate on traps vs on answerable questions
  T4  real vs built split: the memorization check

Why T2 exists: systems that refuse more get judged on fewer, easier
questions, so raw judged means flatter the timid. T2 keeps only the
questions that EVERY system answered, restoring like-for-like.

Why T4 exists: the real (ESMA) questions are public documents a model may
have seen in training; the built questions are new text. A system that
scores far better on real than on built questions of the same kind is
probably remembering, not reading.

USAGE
  python -m evaluation.paper_tables runs/<run_dir>
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

SYSTEMS = ["B0", "B1", "V1", "V2"]
PAPER_LABEL = {"B0": "B0", "B1": "B1", "V1": "MQ", "V2": "AKN"}

CUSTOM = ["CVR", "APA_article", "APA_pinpoint", "MIC", "RC", "CD"]
RAGAS = ["faithfulness", "answer_relevancy", "context_precision",
         "context_recall", "answer_correctness"]


def _fnum(v: str) -> Optional[float]:
    if v in ("", "nan", None):
        return None
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except ValueError:
        return None


def _mean_se(vals: List[float]):
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = sum(vals) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return m, (var / n) ** 0.5, n


def _fmt(m, se, n):
    if m is None:
        return "--"
    return f"{m:.3f} ({n})"


def _console_table(title: str, header: List[str], rows: List[List[str]]):
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    out = [f"\n== {title} =="]
    out.append("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def _latex_table(caption: str, label: str, header: List[str],
                 rows: List[List[str]]) -> str:
    cols = "l" + "c" * (len(header) - 1)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(str(c).replace("_", r"\_") for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run_dir = Path(sys.argv[1])
    mrows = list(csv.DictReader(open(run_dir / "metrics.csv")))
    arows = [json.loads(l) for l in open(run_dir / "answers.jsonl")
             if l.strip()]
    outdir = run_dir / "paper_tables"
    outdir.mkdir(exist_ok=True)

    intent_of = {r["question_id"]: r.get("intent", "") for r in mrows}
    refused = {(r["system_id"], r["question_id"]): r["refused"] == "True"
               for r in mrows}
    qids = sorted({r["question_id"] for r in mrows})

    beyond: set = set()
    esma_path = run_dir.resolve().parent.parent / "data" / "esma_qa.jsonl"
    if esma_path.exists():
        for line in open(esma_path):
            if line.strip():
                q = json.loads(line)
                if q.get("beyond_corpus"):
                    beyond.add(q.get("id"))

    # ---------------- T1: headline deterministic metrics ----------------
    header = ["metric"] + [PAPER_LABEL[s] for s in SYSTEMS]
    rows_out = []
    for m in CUSTOM:
        row = [m]
        for s in SYSTEMS:
            vals = [_fnum(r[m]) for r in mrows if r["system_id"] == s]
            vals = [v for v in vals if v is not None]
            row.append(_fmt(*_mean_se(vals)))
        rows_out.append(row)
    print(_console_table("T1 deterministic metrics, mean (n scored)",
                         header, rows_out))
    (outdir / "t1_deterministic.tex").write_text(_latex_table(
        "Deterministic regulatory metrics, mean over scored answers "
        "(number scored in brackets). MQ cites in prose, so mark based "
        "CVR and CD do not apply to it.",
        "tab:deterministic", header, rows_out))

    # ---------------- T2: judged metrics, common answered subset --------
    common = [q for q in qids
              if all(not refused.get((s, q), True) for s in SYSTEMS)]
    rows_out = []
    for m in RAGAS:
        row = [m]
        for s in SYSTEMS:
            vals = [_fnum(r[m]) for r in mrows
                    if r["system_id"] == s and r["question_id"] in common]
            vals = [v for v in vals if v is not None]
            row.append(_fmt(*_mean_se(vals)))
        rows_out.append(row)
    print(_console_table(
        f"T2 judged metrics on the common answered subset "
        f"(n={len(common)} questions all four systems answered)",
        header, rows_out))
    (outdir / "t2_judged_common.tex").write_text(_latex_table(
        f"LLM-judged metrics restricted to the {len(common)} questions "
        "answered by all four systems, removing the refusal-selection "
        "bias of raw means.",
        "tab:judged", header, rows_out))

    # ---------------- T3: refusal calibration ---------------------------
    rows_out = []
    for s in SYSTEMS:
        traps = [q for q in qids if intent_of[q] == "OFF_TOPIC"]
        ans = [q for q in qids
               if intent_of[q] != "OFF_TOPIC" and q not in beyond]
        r_trap = sum(refused.get((s, q), False) for q in traps)
        r_ans = sum(refused.get((s, q), False) for q in ans)
        rows_out.append([
            PAPER_LABEL[s],
            f"{r_trap}/{len(traps)}" if traps else "--",
            f"{r_ans}/{len(ans)}",
            f"{(r_trap / len(traps)):.2f}" if traps else "--",
            f"{(r_ans / len(ans)):.2f}",
        ])
    header3 = ["system", "refused traps", "refused answerable",
               "trap rate (want 1.0)", "over-refusal (want 0.0)"]
    print(_console_table("T3 refusal calibration", header3, rows_out))
    n_beyond_drawn = len([q for q in qids if q in beyond])
    beyond_note = (
        f" {n_beyond_drawn} question(s) citing provisions inserted after "
        "the 2014 base text are excluded from the answerable set, since "
        "the corpus cannot support them." if n_beyond_drawn else "")
    (outdir / "t3_refusal.tex").write_text(_latex_table(
        "Refusal calibration: refusing every trap and no answerable "
        "question is ideal." + beyond_note,
        "tab:refusal", header3, rows_out))

    # ---------------- T4: real vs built (memorization check) ------------
    is_real = {q: q.startswith("ESMA") for q in qids}
    rows_out = []
    for m in ["answer_correctness", "faithfulness", "APA_article"]:
        for split, keep in (("real", True), ("built", False)):
            row = [f"{m} [{split}]"]
            for s in SYSTEMS:
                vals = [_fnum(r[m]) for r in mrows
                        if r["system_id"] == s
                        and is_real[r["question_id"]] == keep
                        and intent_of[r["question_id"]] == "FACTUAL_QA"]
                vals = [v for v in vals if v is not None]
                row.append(_fmt(*_mean_se(vals)))
            rows_out.append(row)
    print(_console_table(
        "T4 real vs built, fact questions only (memorization check)",
        header, rows_out))
    (outdir / "t4_real_vs_built.tex").write_text(_latex_table(
        "Real (ESMA) versus built questions, fact kind only. A large gap "
        "in favor of real questions suggests memorization of public "
        "documents rather than reading.",
        "tab:realbuilt", header, rows_out))

    print(f"\nLaTeX fragments written to {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
