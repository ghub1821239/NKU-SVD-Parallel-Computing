import re
import csv
from pathlib import Path

RESULT_DIR = Path("./NEW_perf_results")
OUT_CSV = RESULT_DIR / "summary_kernel_time.csv"

pattern_bidiag = re.compile(r"总上二对角化耗时\(ms\):\s*([0-9.]+)")
pattern_gkh = re.compile(r"总GKH迭代耗时\(ms\):\s*([0-9.]+)")

rows = []

for file in sorted(RESULT_DIR.glob("*_program.out")):
    text = file.read_text(encoding="utf-8", errors="ignore")

    bidiag_times = [float(x) for x in pattern_bidiag.findall(text)]
    gkh_times = [float(x) for x in pattern_gkh.findall(text)]

    if not bidiag_times or not gkh_times:
        print(f"Skip {file.name}: no timing data found")
        continue

    n = min(len(bidiag_times), len(gkh_times))

    bidiag_times = bidiag_times[:n]
    gkh_times = gkh_times[:n]
    total_times = [b + g for b, g in zip(bidiag_times, gkh_times)]

    rows.append({
        "version": file.name.replace("_program.out", ""),
        "runs": n,
        "avg_bidiag_ms": sum(bidiag_times) / n,
        "avg_gkh_ms": sum(gkh_times) / n,
        "avg_kernel_total_ms": sum(total_times) / n,
        "min_kernel_total_ms": min(total_times),
        "max_kernel_total_ms": max(total_times),
    })

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "version",
        "runs",
        "avg_bidiag_ms",
        "avg_gkh_ms",
        "avg_kernel_total_ms",
        "min_kernel_total_ms",
        "max_kernel_total_ms",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {OUT_CSV}")