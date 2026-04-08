import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 11

DATA_PATH = '../data/product_state_varcov_v2.npz'

def main():
    data = np.load(DATA_PATH, allow_pickle=True)

    ny_values = list(range(1, 11))
    n_qubits = [2 * ny for ny in ny_values]  # nx=2, spinless

    variances = []
    covariances = []
    for ny in ny_values:
        d = data[f'ny{ny}']
        variances.append(d[:, 0])
        covariances.append(d[:, 1])

    var_means = [v.mean() for v in variances]

    fig, ax = plt.subplots(figsize=(10, 6))

    parts = ax.violinplot(covariances, positions=n_qubits, widths=1.5,
                          showmedians=True, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#ff9f43')
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('#e17055')
    parts['cmedians'].set_linewidth(2)

    ax.plot(n_qubits, var_means, 's-', color='#0984e3', linewidth=2,
            markersize=8, label=r'$\sum_i c_i^2 \mathrm{Var}(P_i)$', zorder=5)

    from matplotlib.patches import Patch
    violin_patch = Patch(facecolor='#ff9f43', alpha=0.7,
                         label=r'$2\sum_{i<j} c_i c_j \mathrm{Cov}(P_i, P_j)$')
    ax.legend(handles=[ax.get_lines()[0], violin_patch])

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Number of qubits')
    ax.set_ylabel(r'Energy$^2$')
    ax.set_xticks(n_qubits)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../output/product_state_varcov.png', dpi=150, bbox_inches='tight')
    plt.savefig('../output/product_state_varcov.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: ../output/product_state_varcov.{png,pdf}')

if __name__ == '__main__':
    main()
