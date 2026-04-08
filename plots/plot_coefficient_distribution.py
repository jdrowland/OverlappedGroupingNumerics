import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import openfermion as of
from ogn.loaders import HDF5Loader
from ogn.sorted_insertion import sorted_insertion_grouping
from ogn.adhoc_repacking import adhoc_repacking

OUTPUT_DIR = Path(__file__).parent / '../output'

MOLECULE_PATH = str(Path(of.__file__).parent / 'testing' / 'data' / 'H2-O1_sto-3g_singlet_H2O.hdf5')

def compute_group_coefficient_sums(groups):
    sums = []
    for group in groups.groups:
        group_sum = sum(abs(float(np.real(p.coeff))) for p in group.paulis)
        sums.append(group_sum)
    return sums

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    loader = HDF5Loader(MOLECULE_PATH, hermitianize=True)
    hamiltonian = loader.load()

    si_groups = sorted_insertion_grouping(hamiltonian)
    adhoc_groups = adhoc_repacking(hamiltonian, si_groups)

    si_sums = sorted(compute_group_coefficient_sums(si_groups), reverse=True)
    adhoc_sums = sorted(compute_group_coefficient_sums(adhoc_groups), reverse=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(range(1, len(si_sums) + 1), si_sums, 'o-', label='Sorted Insertion',
                color='#1f77b4', markersize=4, linewidth=1.5)
    ax.semilogy(range(1, len(adhoc_sums) + 1), adhoc_sums, 's-', label='Ad-hoc Repacking',
                color='#2ca02c', markersize=4, linewidth=1.5)

    ax.set_xlabel("Group Index", fontsize=18)
    ax.set_ylabel(r"$\sum_i |c_i|$ per group", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=15)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'H2O_coefficient_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'H2O_coefficient_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: output/H2O_coefficient_distribution.{png,pdf}")

if __name__ == "__main__":
    main()
