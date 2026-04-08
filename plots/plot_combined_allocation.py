import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 11

MOLECULES = ['BeH2', 'H6', 'LiH', 'H2O', 'NH3', 'CH4', 'OWP']
STATES = ['dmrg', 'hf', 'random']
RESULTS_DIR = Path(__file__).parent / '../data/results_optimal'
OUTPUT_DIR = Path(__file__).parent / '../output'

MOL_LABELS = {
    'BeH2': r'BeH$_2$', 'H6': r'H$_6$', 'LiH': 'LiH',
    'H2O': r'H$_2$O', 'NH3': r'NH$_3$', 'CH4': r'CH$_4$', 'OWP': 'OWP',
}

COLOR_ADHOC_SI = '#ff7f0e'
COLOR_ADHOC = '#2ecc71'
COLOR_POSTHOC = '#3498db'


def load_results():
    results = {}
    for mol in MOLECULES:
        results[mol] = {}
        for state in STATES:
            results[mol][state] = {}
            for method in ['sorted_insertion', 'adhoc_si_allocation', 'adhoc_repacking', 'posthoc_repacking']:
                fname = RESULTS_DIR / f"{mol}_{state}_{method}_optimal_result.json"
                if fname.exists():
                    with open(fname) as f:
                        results[mol][state][method] = json.load(f)
    return results


def get_labels(results, state):
    labels = []
    for mol in MOLECULES:
        name = MOL_LABELS[mol]
        if mol in results and state in results[mol] and 'sorted_insertion' in results[mol][state]:
            n_groups = results[mol][state]['sorted_insertion']['num_groups']
            labels.append(f"{name}\n({n_groups})")
        else:
            labels.append(name)
    return labels


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = load_results()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    x = np.arange(len(MOLECULES))
    bar_width = 0.25

    for ax_idx, (ax, state) in enumerate(zip(axes, STATES)):
        adhoc_si_ratios, adhoc_opt_ratios, posthoc_ratios = [], [], []
        for mol in MOLECULES:
            ms = results.get(mol, {}).get(state, {})
            si_var = ms.get('sorted_insertion', {}).get('total_variance', 0)
            adhoc_si_ratios.append(si_var / ms['adhoc_si_allocation']['total_variance'] if ms.get('adhoc_si_allocation') and si_var else 0)
            adhoc_opt_ratios.append(si_var / ms['adhoc_repacking']['total_variance'] if ms.get('adhoc_repacking') and si_var else 0)
            posthoc_ratios.append(si_var / ms['posthoc_repacking']['total_variance'] if ms.get('posthoc_repacking') and si_var else 0)

        ax.bar(x - bar_width, adhoc_si_ratios, bar_width, label='Ad-hoc + SI Alloc.', color=COLOR_ADHOC_SI, edgecolor='black', linewidth=0.5)
        ax.bar(x, adhoc_opt_ratios, bar_width, label='Ad-hoc + Opt Alloc.', color=COLOR_ADHOC, edgecolor='black', linewidth=0.5)
        ax.bar(x + bar_width, posthoc_ratios, bar_width, label='Post-hoc Repacking', color=COLOR_POSTHOC, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(get_labels(results, state))
        ax.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel(r'Variance Reduction ($\sigma^2_{\mathrm{SI}} / \sigma^2_{\mathrm{repacked}}$)')

    axes[0].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'combined_allocation_reduction.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'combined_allocation_reduction.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: output/combined_allocation_reduction.{png,pdf}")

if __name__ == "__main__":
    main()
