import re
import csv
from pathlib import Path
RESULT_DIR = Path("./NEW_perf_results")
OUT_CSV = RESULT_DIR / "summary_memory_bound.csv"

patterns = {
    "cpu_cycles": re.compile(r"^\s*([\d,]+)\s+cpu_cycles:u", re.MULTILINE),
    "mem_stall_anyload": re.compile(r"^\s*([\d,]+)\s+mem_stall_anyload:u", re.MULTILINE),
    "mem_stall_l1miss": re.compile(r"^\s*([\d,]+)\s+mem_stall_l1miss:u", re.MULTILINE),
    "mem_stall_l2miss": re.compile(r"^\s*([\d,]+)\s+mem_stall_l2miss:u", re.MULTILINE),
}

def parse_int(s):
    return int(s.replace(",", ""))

rows = []

for file in sorted(RESULT_DIR.glob("*_perf_stat.txt")):
    text = file.read_text(encoding="utf-8", errors="ignore")

    values = {}

    for name, pattern in patterns.items():
        matches = pattern.findall(text)
        if not matches:
            print(f"Skip {file.name}: missing {name}")
            values[name] = None
        else:
            # perf stat 里 cpu_cycles / mem_stall 可能重复出现，取第一次即可
            values[name] = parse_int(matches[0])

    if any(v is None for v in values.values()):
        continue

    cpu_cycles = values["cpu_cycles"]
    mem_stall_anyload = values["mem_stall_anyload"]
    mem_stall_l1miss = values["mem_stall_l1miss"]
    mem_stall_l2miss = values["mem_stall_l2miss"]

    l1_bound = (mem_stall_anyload - mem_stall_l1miss) / cpu_cycles
    l2_bound = (mem_stall_l1miss - mem_stall_l2miss) / cpu_cycles
    mem_bound = mem_stall_l2miss / cpu_cycles

    rows.append({
        "version": file.name.replace("_perf_stat.txt", ""),
        "cpu_cycles": cpu_cycles,
        "mem_stall_anyload": mem_stall_anyload,
        "mem_stall_l1miss": mem_stall_l1miss,
        "mem_stall_l2miss": mem_stall_l2miss,
        "l1_bound": l1_bound,
        "l2_bound": l2_bound,
        "mem_bound": mem_bound,
        "l1_bound_percent": l1_bound * 100,
        "l2_bound_percent": l2_bound * 100,
        "mem_bound_percent": mem_bound * 100,
    })

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "version",
        "cpu_cycles",
        "mem_stall_anyload",
        "mem_stall_l1miss",
        "mem_stall_l2miss",
        "l1_bound",
        "l2_bound",
        "mem_bound",
        "l1_bound_percent",
        "l2_bound_percent",
        "mem_bound_percent",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {OUT_CSV}")