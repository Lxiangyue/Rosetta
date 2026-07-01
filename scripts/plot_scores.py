import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt

RUNS = [
    ("Rosetta", "rosetta"),
    ("MoE", "moe"),
    ("MoT", "mot"),
]
STEPS = [0, 100]


def read_arc_score(run_dir: Path, step: int) -> float:
    result_path = (
        run_dir
        / "eval_outputs"
        / f"{step:07d}"
        / "arc_challenge__arc_challenge_1"
        / "metric_results"
        / "arc_challenge.json"
    )
    if not result_path.exists():
        raise FileNotFoundError(f"Missing ARC result: {result_path}")

    data = json.loads(result_path.read_text())
    for item in reversed(data):
        metric = str(item.get("metric", "")).lower()
        if "avg" in metric and item.get("value") is not None:
            return float(item["value"]) * 100.0
    for item in reversed(data):
        if item.get("value") is not None:
            return float(item["value"]) * 100.0
    raise ValueError(f"No ARC score found in {result_path}")


def collect_scores(run_dir: Path):
    scores = {}
    rows = []
    for model_name, run_name in RUNS:
        model_run_dir = run_dir / run_name
        scores[model_name] = []
        for step in STEPS:
            score = read_arc_score(model_run_dir, step)
            scores[model_name].append(score)
            rows.append({
                "model": model_name,
                "step": step,
                "arc_challenge": f"{score:.2f}",
            })
    return scores, rows


def save_csv(rows, out_dir: Path) -> Path:
    csv_path = out_dir / "arc_step0_step100.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "step", "arc_challenge"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_scores(scores, out_dir: Path) -> Path:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         17,
        "axes.labelsize":    19,
        "axes.titlesize":    18,
        "axes.labelweight":  "bold",
        "xtick.labelsize":   16,
        "ytick.labelsize":   16,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.major.size":  5.0,
        "ytick.major.size":  5.0,
        "axes.linewidth":    1.8,
        "legend.fontsize":   14,
        "legend.framealpha": 0.95,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    colors = {
        "Rosetta": "#C0392B",  # red
        "MoE":     "#4472C4",  # blue
        "MoT":     "#E69F00",  # orange
    }
    markers = {
        "Rosetta": "o",
        "MoE":     "s",
        "MoT":     "v",
    }

    for model_name, values in scores.items():
        color = colors.get(model_name, "#999999")
        marker = markers.get(model_name, "D")
        ax.plot(STEPS, values,
                marker=marker, linewidth=3.5, markersize=10,
                color=color, label=model_name, linestyle="-",
                markerfacecolor=color, markeredgewidth=0,
                markeredgecolor=color, alpha=0.95, zorder=10)

    sorted_step0 = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    for rank, (model_name, values) in enumerate(sorted_step0):
        color = colors.get(model_name, "#999999")
        val_0 = values[0]
        if rank == 0:
            xytext = (-8, 8)
            ha = "right"
        elif rank == 1:
            xytext = (8, 8)
            ha = "left"
        else:
            xytext = (0, -18)
            ha = "center"
        ax.annotate(f"{val_0:.1f}", (0, val_0), textcoords="offset points",
                    xytext=xytext, ha=ha, fontsize=13, color=color, fontweight="bold")

    for model_name, values in scores.items():
        color = colors.get(model_name, "#999999")
        val_100 = values[1]
        ax.annotate(f"{val_100:.1f}", (100, val_100), textcoords="offset points",
                    xytext=(10, -4), ha="left", fontsize=13, color=color, fontweight="bold")

    # ax.set_title("ARC-Challenge after 100-step training", fontsize=17, pad=10, fontweight="bold")
    ax.set_ylabel("ARC-Challenge Score (%)", fontsize=18, fontweight="bold", labelpad=8, color="#222222")
    ax.set_xlabel("Training step", fontsize=17, fontweight="bold", labelpad=0, color="#222222")

    ax.set_xticks(STEPS)
    ax.set_xticklabels([f"{s}" for s in STEPS], fontsize=15, fontweight="bold")
    ax.set_xlim(-15, 115)

    all_scores = [score for values in scores.values() for score in values]
    y_min = max(0.0, min(all_scores) - 10.0)
    y_max = min(100.0, max(all_scores) + 8.0)
    ax.set_ylim(y_min, y_max)

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(15)

    ax.grid(True, axis="y", linestyle="--", alpha=0.30, linewidth=0.9, color="#BBBBBB")
    ax.set_axisbelow(True)

    legend = ax.legend(
        loc="lower left", frameon=True, fancybox=False, ncol=1,
        handlelength=1.8, handletextpad=0.5, borderpad=0.5, labelspacing=0.4,
        edgecolor="#E0E0E0"
    )
    legend.get_frame().set_linewidth(1.0)
    for line in legend.get_lines():
        line.set_linewidth(3.0)

    plt.tight_layout(pad=1.2)
    png_path = out_dir / "arc_step0_step100.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    return png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/example_train")
    parser.add_argument("--out-dir", default="outputs/example_train")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores, rows = collect_scores(run_dir)
    csv_path = save_csv(rows, out_dir)
    png_path = plot_scores(scores, out_dir)
    print(f"[plot_scores] Saved CSV: {csv_path}")
    print(f"[plot_scores] Saved plot: {png_path}")


if __name__ == "__main__":
    main()
