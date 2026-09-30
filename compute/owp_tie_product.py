import sys
import json
import time
import pickle
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ogn.pauli import PauliString
from ogn.group import PauliGroup, GroupCollection
from owp_product_state import variance, product_state

N_QUBITS = 44
TOTAL_SHOTS = 100_000


def si_allocation(arrs):
    # sigma_g = sqrt(sum_{i in g, P_i != I} c_i^2); shots proportional to sigma_g
    # (compute_si_sigma_state_independent + compute_optimal_allocation in run_optimal_allocation.py)
    nonid = (arrs['x'] != 0) | (arrs['z'] != 0)
    g = np.repeat(np.arange(len(arrs['sizes'])), arrs['sizes'])
    sig = np.sqrt(np.bincount(g, weights=np.where(nonid, arrs['c'] ** 2, 0.0), minlength=len(arrs['sizes'])))
    return TOTAL_SHOTS * sig / sig.sum()


def optimized_allocation(arrs, warm):
    gc, k = GroupCollection(), 0
    for s in arrs['sizes']:
        pg = PauliGroup()
        for i in range(k, k + s):
            pg.add(PauliString(int(arrs['x'][i]), int(arrs['z'][i]), float(arrs['c'][i]), N_QUBITS))
        gc.groups.append(pg); k += s
    shots, _ = gc.shot_count_optimized(TOTAL_SHOTS, warm_start=np.asarray(warm))
    return np.asarray(shots)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')        # groupings_seed*_sample*.pkl from owp_tie_groupings.py
    p.add_argument('out')
    args = p.parse_args()
    t0 = time.time()
    g = pickle.load(open(args.path, 'rb'))
    G = g['groupings']
    si = si_allocation(G['sorted_insertion'])
    opt = optimized_allocation(G['adhoc'], si)
    np.savez(Path(args.out).with_suffix('.shots.npz'), si_shots=si, opt_shots=opt)
    cases = {'sorted_insertion': (G['sorted_insertion'], si), 'adhoc_si_allocation': (G['adhoc'], si),
             'adhoc_repacking': (G['adhoc'], opt), 'posthoc_repacking': (G['posthoc'], si)}
    res = {'seed': g['seed'], 'num_groups': int(len(G['sorted_insertion']['sizes']))}
    for st in ('hf', 'random'):
        state = product_state(st)
        res[st] = {m: float(variance(a, s, state)) for m, (a, s) in cases.items()}
    res['seconds'] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
