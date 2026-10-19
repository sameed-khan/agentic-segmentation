"""Compare Phase 3 / 3b test-split runs (169 cases): per-histotype mean Dice vs U-Net, gate, loc. fails, tools,
tokens, cost. Usage: uv run --project harness/server python eval/v2/sweep_summary.py [run ...] (default: all)"""
import csv, json, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
U = {"F": 0.507, "L": 0.744, "M": 0.823, "N": 0.748, "V": 0.584}
runs = sys.argv[1:] or sorted(p.name for p in (ROOT / "results").iterdir()
                             if (p.name.startswith("p3-") or p.name.startswith("phase3-")) and (p / "scores.csv").exists())
print("| run | F | L | M | N | V | unweighted | median | >U-Net | loc fails | forced | tools | prompts | out tok/case | cost USD |")
print("|---|" + "---|" * 14)
print("| published U-Net | " + " | ".join(f"{U[h]:.3f}" for h in U) + " | 0.681 | | | | | | | | |")
for run in runs:
    r = list(csv.DictReader(open(ROOT / "results" / run / "scores.csv")))
    if len(r) != 169:
        print(f"| {run} (n={len(r)}, incomplete) |"); continue
    hm = {h: st.mean(float(x["dice"]) for x in r if x["histotype"] == h) for h in U}
    print(f"| {run} | " + " | ".join(f"{hm[h]:.3f}" for h in U) +
          f" | {st.mean(hm.values()):.3f} | {st.median(float(x['dice']) for x in r):.3f} | {sum(hm[h] > U[h] for h in U)} | "
          f"{sum(x['tag'] == 'localization' for x in r)} | {sum(x['forced_finish'] in ('True', 'true') for x in r)} | "
          f"{st.mean(int(x['n_tool_calls']) for x in r):.1f} | {st.mean(int(x['n_model_prompts']) for x in r):.2f} | "
          f"{st.mean(int(x['tokens_out'] or 0) for x in r):.0f} | {sum(float(x['cost_usd'] or 0) for x in r):.2f} |")
