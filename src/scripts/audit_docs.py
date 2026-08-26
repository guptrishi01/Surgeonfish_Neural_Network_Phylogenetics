"""Audits documentation against the data it describes.

Run:  python src/scripts/audit_docs.py

`tests/test_published_results.py` already enforces that the result files are
internally consistent. What a test suite structurally cannot catch is *prose*
drifting from the numbers it describes, or two sections of a document
contradicting each other — which is exactly how a sentence calling the
patternize port "the largest unquantified risk" survived several versions past
that risk being closed, sitting directly beneath a table marking it done.

Every individual claim in this project has been true when written. Staleness
comes from the world moving on around it, so these checks compare documents
against the current data and against each other.
"""

import csv
import io
import re
import subprocess
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(f"{name}{': ' + detail if detail else ''}")


def read_csv(rel):
    with open(ROOT / rel, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


readme = (ROOT / "README.md").read_text(encoding="utf-8")
methods = (ROOT / "METHODS.md").read_text(encoding="utf-8")
changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")

print("=" * 72)
print("A. DOC-VS-DATA: do quoted numbers match the result files?")
print("=" * 72)

kmult = {r["dimension"]: r for r in read_csv("outputs/phase4/kmult_results.csv")}
mantel = {r["dimension"]: r for r in read_csv("outputs/phase4/mantel_results.csv")}

for dim in ("color", "stripe", "spot"):
    p = f"{float(kmult[dim]['bh_corrected_p']):.3f}".rstrip("0").rstrip(".")
    check(f"README quotes Kmult {dim} p={p}", p in readme)
for dim in ("color", "stripe", "spot"):
    p = f"{float(mantel[dim]['bh_corrected_p']):.3f}".rstrip("0").rstrip(".")
    check(f"README quotes Mantel {dim} p={p}", p in readme)

agg = read_csv("reports/species_features.csv")
check("README quotes the 49-species analysis set", "49 species" in readme,
      f"{len(agg)} in file")
check("METHODS quotes 648 images", "648" in methods,
      f"{sum(int(r['n_images']) for r in agg)} in file")

pat = read_csv("outputs/patternize_check/equivalence_result.csv")
worst = max(float(r["max_abs_difference"]) for r in pat)
check(f"README quotes patternize worst diff {worst}", str(worst) in readme)

print()
print("=" * 72)
print("B. STALENESS: claims contradicting closed work")
print("=" * 72)

# Follow-up rows marked done must not also be described as open elsewhere.
done_items = re.findall(r"\|\s*(\d)\s*\|\s*~~([^~]+)~~\s*—\s*\*\*done", readme)
check("follow-up table marks closed items with strikethrough",
      len(done_items) >= 3, f"{len(done_items)} marked done")

check("no 'remains the largest unquantified risk' after patternize closed",
      "remains the largest unquantified risk" not in readme)
# Fails only if the row is listed as open (no strikethrough / not marked done).
check("stripe recall row is marked done, not open",
      bool(re.search(r"~~Improve `stripe_present` recall~~", readme)))
check("no 'Kmult re-run pending' language",
      "re-run pending" not in readme and "*re-run pending*" not in readme)
check("no 'not yet run' in README/METHODS",
      "not yet run" not in readme and "not yet run" not in methods)
# Google Colab is the only execution target - CLAUDE.md's code conventions say
# so explicitly - but a docstring claimed the opposite for five months, naming
# an HPC/SLURM setup that was never used and citing CLAUDE.md as its source.
slurm_claims = [
    str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")
    # This file is exempt: stating the rule requires naming what it forbids.
    if p.name != "audit_docs.py"
    and re.search(r"slurm|\bhpc\b", p.read_text(encoding="utf-8"), re.I)
]
check("no source file claims an HPC/SLURM execution target",
      not slurm_claims, ", ".join(slurm_claims))
check("README version matches newest changelog entry",
      (m := re.search(r"\*\*Version (\d+\.\d+\.\d+)\*\*", readme))
      and (c := re.search(r"- \*\*(\d+\.\d+\.\d+)\*\*", changelog))
      and m.group(1) == c.group(1),
      f"README {m.group(1) if m else '?'} vs CHANGELOG {c.group(1) if c else '?'}")

print()
print("=" * 72)
print("C. REPO HEALTH")
print("=" * 72)

status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                        capture_output=True, text=True).stdout.strip()
check("working tree clean", not status, status[:120])

unpushed = subprocess.run(["git", "-C", str(ROOT), "log", "origin/main..HEAD",
                           "--oneline"], capture_output=True, text=True).stdout.strip()
check("nothing unpushed", not unpushed, unpushed[:120])

tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                       capture_output=True, text=True).stdout
check("test suite green", "passed" in tests and "failed" not in tests,
      tests.strip().splitlines()[-1] if tests else "")

lint = subprocess.run([sys.executable, "-m", "ruff", "check", "src", "tests"],
                      cwd=ROOT, capture_output=True, text=True)
check("ruff clean", lint.returncode == 0, lint.stdout.strip()[:100])

# Test counts quoted in prose drift whenever a test is added or un-skipped -
# which is exactly what happened when the comparison test stopped skipping.
actual = re.search(r"(\d+) passed", tests)
if actual:
    n = actual.group(1)
    quoted = set(re.findall(r"(\d{3}) tests", readme))
    check(f"README test counts match the suite ({n})",
          not quoted or quoted == {n}, f"README says {sorted(quoted)}, suite has {n}")

# Every tracked output referenced by a doc must exist, and vice versa: every
# tracked result dir should be explained somewhere.
for d in ("outputs/phase4", "outputs/phase4_min5", "outputs/phase4_noprops",
          "outputs/phase4_pre_recalibration", "outputs/patternize_check",
          "outputs/sensitivity_min5", "outputs/sensitivity_with_reference"):
    name = Path(d).name
    explained = (name in readme or name in methods or name in changelog
                 or (ROOT / d / "README.md").exists())
    check(f"{d} is explained somewhere", explained)

# Orphan check: tracked files nothing references.
scripts = list((ROOT / "src/scripts").glob("*.py"))
figures_readme = (ROOT / "figures/README.md").read_text(encoding="utf-8")
for s in scripts:
    referenced = any(s.name in doc for doc in
                     (readme, methods, changelog, claude, figures_readme))
    check(f"src/scripts/{s.name} is referenced", referenced)

print()
print("=" * 72)
print("D. FIGURES")
print("=" * 72)

figs = sorted((ROOT / "figures").glob("*.png"))
check("five figures present", len(figs) == 5, f"{len(figs)} found")
for f in figs:
    check(f"{f.name} referenced in a doc",
          f.name in readme or f.name in (ROOT / "figures/README.md").read_text(encoding="utf-8"))
    check(f"{f.name} is non-trivial in size", f.stat().st_size > 20_000,
          f"{f.stat().st_size} bytes")

print()
print("=" * 72)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
print("=" * 72)
if FAIL:
    print("\nFAILURES:")
    for f in FAIL:
        print("  X", f)
else:
    print("\nNo issues found.")
