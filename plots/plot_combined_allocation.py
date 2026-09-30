import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 11

MOLECULES = ['BeH2', 'H6', 'LiH', 'H2O', 'NH3', 'CH4', 'OWP']
STATES = ['dmrg', 'hf', 'random']
N_SAMPLES = 10
RESULTS_DIR = Path(__file__).parent / '../data/results_optimal'
OUTPUT_DIR = Path(__file__).parent / '../output'

MOL_LABELS = {
    'BeH2': r'BeH$_2$', 'H6': r'H$_6$', 'LiH': 'LiH',
    'H2O': r'H$_2$O', 'NH3': r'NH$_3$', 'CH4': r'CH$_4$', 'OWP': 'OWP',
}
METHODS = [
    (-0.25, 'adhoc_si_allocation', '#ff7f0e', 'Ad-hoc + SI Alloc.'),
    (0.0, 'adhoc_repacking', '#2ecc71', 'Ad-hoc + Opt Alloc.'),
    (0.25, 'posthoc_repacking', '#3498db', 'Post-hoc Repacking'),
]


def load(mol, state):
    path = RESULTS_DIR / f'{mol}_{state}_tiesamples.json'
    if not path.exists():
        sys.exit(f"missing {path}")
    samples = json.load(open(path))['samples']
    if len(samples) != N_SAMPLES:
        sys.exit(f"{path}: {len(samples)} tie samples, expected {N_SAMPLES}")
    return samples


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), sharey=True)
    x = np.arange(len(MOLECULES))
    for ax, state, letter in zip(axes, STATES, 'abc'):
        samples = {mol: load(mol, state) for mol in MOLECULES}
        for off, method, color, label in METHODS:
            r = [np.array([s['sorted_insertion'] / s[method] for s in samples[mol]]) for mol in MOLECULES]
            med = np.array([np.median(v) for v in r])
            err = [med - np.array([v.min() for v in r]), np.array([v.max() for v in r]) - med]
            ax.bar(x + off, med, 0.25, color=color, edgecolor='black', linewidth=0.5, label=label)
            ax.errorbar(x + off, med, yerr=err, fmt='none', ecolor='black', elinewidth=0.8, capsize=2)
        groups = {mol: [s['num_groups'] for s in samples[mol]] for mol in MOLECULES}
        ax.set_xticks(x)
        ax.set_xticklabels([f"{MOL_LABELS[m]}\n{min(groups[m])}–{max(groups[m])}" for m in MOLECULES], fontsize=9.5)
        ax.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel(r'Variance Reduction ($\sigma^2_{\mathrm{SI}} / \sigma^2_{\mathrm{repacked}}$)')
    axes[0].legend(loc='upper left')
    plt.tight_layout()
    stem = 'combined_allocation_reduction'
    for ext in ('png', 'pdf'):
        plt.savefig(OUTPUT_DIR / f'{stem}.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: output/{stem}.{{png,pdf}}")


if __name__ == '__main__':
    main()
