import sys
import json
import time
import pickle
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from lookup_variance import sorted_table, variance_lookup_multi

N_QUBITS = 44


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')              # ogn_ties/out: batchA/, batchB/ (keys + vals_all), groupings, tie_product/
    p.add_argument('owp_npz')          # data/owp/dmrg_expectations.npz (single-term <P>)
    p.add_argument('samples', type=lambda v: [int(k) for k in v.split(',')])
    args = p.parse_args()
    out = Path(args.out)

    tables = []
    for b in ('batchA', 'batchB'):
        kx = np.load(out / b / 'keys_x.npy')
        kz = np.load(out / b / 'keys_z.npy')
        kv = np.load(out / b / 'vals_all.npy')
        tables.append((kx, kz, kv))
    d = np.load(args.owp_npz)
    tables.append(sorted_table(d['x_bits'].astype(np.int64), d['z_bits'].astype(np.int64), d['exp_vals']))

    for sample in args.samples:
        run_sample(out, sample, tables)


def run_sample(out, sample, tables):
    t0 = time.time()
    g = pickle.load(open(out / f'groupings_seed0_sample{sample}.pkl', 'rb'))
    G = g['groupings']
    sh = np.load(out / 'tie_product' / f'OWP_sample{sample}.shots.npz')
    si, opt = sh['si_shots'], sh['opt_shots']
    cases = {'sorted_insertion': (G['sorted_insertion'], si), 'adhoc_si_allocation': (G['adhoc'], si),
             'adhoc_repacking': (G['adhoc'], opt), 'posthoc_repacking': (G['posthoc'], si)}
    res = {'seed': g['seed'], 'num_groups': int(len(G['sorted_insertion']['sizes']))}
    for m, (arrs, shots) in cases.items():
        t = time.time()
        res[m] = float(variance_lookup_multi(arrs, shots, tables, N_QUBITS, parallel=True))
        print(f"sample {sample} {m}: {res[m]:.12e} ({time.time() - t:.0f}s)", flush=True)
    res['seconds'] = round(time.time() - t0, 1)
    (out / 'tie_product' / f'OWP_sample{sample}.dmrg.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
