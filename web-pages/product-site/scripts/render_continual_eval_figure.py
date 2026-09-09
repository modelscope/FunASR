"""Reproduce the synthetic article figure with matplotlib==3.10.0 (not model results)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SITE = Path(__file__).resolve().parents[1]
data = json.loads((SITE / "data/continual-eval-example.json").read_text())
rows = [row for row in data["slices"] if row["old_domain"]]
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
for offset, key, color, name in [(-0.16, "a_errors", "#c53a46", "Candidate A"),
                                 (0.16, "b_errors", "#147566", "Candidate B")]:
    changes = [100 * (row[key] - row["baseline_errors"]) / row["reference_characters"] for row in rows]
    y = [i + offset for i in range(len(rows))]
    ax.barh(y, changes, height=0.27, color=color, label=name)
    for yi, change in zip(y, changes):
        ax.text(max(0, change) + 0.08, yi, f"{change:+.1f} pp", va="center", fontsize=11, color="#202827")
ax.axvline(data["old_domain_limit_pp"], color="#656e68", linestyle="--", linewidth=1.4)
ax.text(1.1, -0.63, "Illustrative limit: +1 pp", fontsize=11, color="#444c48")
ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
ax.invert_yaxis()
ax.set_xlim(-1.5, 7.4)
ax.set_xlabel("Change in validation CER from the original model (percentage points)")
ax.set_title("A better overall score can hide a worse old domain", loc="left", pad=26, fontsize=16)
ax.legend(loc="upper right", frameon=False)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.text(0.02, 0.01, "SYNTHETIC EXAMPLE. No model was measured. Limits are not FunASR recommendations.", fontsize=10)
fig.subplots_adjust(left=0.28, right=0.97, bottom=0.22, top=0.80)
output = SITE / "legacy/img/continual-eval-example.png"
fig.savefig(output, dpi=140, metadata={"Software": "FunASR synthetic editorial example"})
plt.close(fig)
print(output)
