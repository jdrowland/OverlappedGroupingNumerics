import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 11

RESULTS_DIR = Path(__file__).parent / '../data/results_scaling'
OUTPUT_DIR = Path(__file__).parent / '../output'
N_QUBITS = list(range(4, 13))

METHODS = ["adhoc_si", "adhoc_opt", "posthoc"]
METHOD_LABELS = ["Ad-hoc + SI", "Ad-hoc + Opt", "Post-hoc"]
METHOD_COLORS = ["#ff7f0e", "#2ca02c", "#d62728"]
METHOD_MARKERS = ["s", "^", "D"]

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = {}
    for n in N_QUBITS:
        fname = RESULTS_DIR / f"result_{n}q_all_methods_cov.json"
        if fname.exists():
            with open(fname) as f:
                results[n] = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, label, color, marker in zip(METHODS, METHOD_LABELS, METHOD_COLORS, METHOD_MARKERS):
        ratios, qubits = [], []
        rkey = f"ratio_{method}"
        for n in N_QUBITS:
            if n in results and rkey in results[n] and results[n][rkey] and results[n][rkey] > 0:
                ratios.append(results[n][rkey])
                qubits.append(n)
        if ratios:
            ax.plot(qubits, ratios, marker=marker, label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Number of Qubits (Number of Groups)")
    ax.set_ylabel(r"Variance Reduction ($\sigma^2_{\mathrm{SI}}/\sigma^2_{\mathrm{repacked}}$)")
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(N_QUBITS)

    xlabels = []
    for n in N_QUBITS:
        if n in results:
            xlabels.append(f"{n}\n({results[n]['num_groups']:,})")
        else:
            xlabels.append(str(n))
    ax.set_xticklabels(xlabels)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'variance_reduction_scaling_cov.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'variance_reduction_scaling_cov.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: output/variance_reduction_scaling_cov.{png,pdf}")

if __name__ == "__main__":
    main()
