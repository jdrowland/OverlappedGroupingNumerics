import re
import json
import glob
import argparse
import platform
from pathlib import Path

import numpy as np
import numba
import scipy

STATES = ('dmrg', 'hf', 'random')
N_SAMPLES = 10


def main():
    p = argparse.ArgumentParser()
    p.add_argument('tie_product_dir')   # owp_tie_product.py and owp_dmrg_variances.py outputs
    p.add_argument('dest_dir')          # data/results_optimal
    args = p.parse_args()
    src, dest = Path(args.tie_product_dir), Path(args.dest_dir)
    ks = sorted(int(m.group(1)) for f in glob.glob(str(src / 'OWP_sample*.json'))
                if (m := re.search(r'OWP_sample(\d+)\.json$', f)))
    if ks != list(range(N_SAMPLES)):
        raise SystemExit(f"expected samples 0..{N_SAMPLES - 1}, found {ks}")
    prov = {'host': platform.node(), 'python': platform.python_version(),
            'versions': {'numpy': np.__version__, 'numba': numba.__version__, 'scipy': scipy.__version__},
            'hamiltonian': 'data/owp/dmrg_expectations.npz', 'mps': 'data/owp/owp_reactant_chi64.pkl (DMRG chi=64)',
            'groupings': 'compute/owp_tie_groupings.py, seed [0, k]', 'n_electrons_hf': 32, 'random_state_seed': 42,
            'total_shots': 100000,
            'note': 'shot_count_optimized reaches scipy maxiter=500 at OWP scale without converging; '
                    'effect on the v1-grouping variance <= 3.4e-5 relative'}
    for st in STATES:
        if st == 'dmrg' and not all((src / f'OWP_sample{k}.dmrg.json').exists() for k in ks):
            print(f"OWP dmrg: skipped, DMRG results missing for some of samples {ks}")
            continue
        samples = []
        for k in ks:
            if st == 'dmrg':
                r = json.load(open(src / f'OWP_sample{k}.dmrg.json'))
                if r['seed'] != [0, k]:
                    raise SystemExit(f"sample {k}: seed {r['seed']} != [0, {k}]")
                samples.append({m: r[m] for m in ('seed', 'num_groups', 'sorted_insertion', 'adhoc_si_allocation',
                                                   'adhoc_repacking', 'posthoc_repacking')})
            else:
                r = json.load(open(src / f'OWP_sample{k}.json'))
                samples.append(dict(seed=r['seed'], num_groups=r['num_groups'], **r[st]))
        out = {'molecule': 'OWP', 'state_type': st, 'n_qubits': 44, 'n_terms': 575711, 'dmrg_energy': None,
               'tie_decimals': 10, 'tie_samples': len(samples), 'seed': 0, 'samples': samples, 'provenance': prov}
        (dest / f'OWP_{st}_tiesamples.json').write_text(json.dumps(out, indent=2))
        print(f"OWP {st}: {len(samples)} samples -> {dest / f'OWP_{st}_tiesamples.json'}")


if __name__ == '__main__':
    main()
