import os
import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pauli_sweep import mps_site_arrays, batch_expectations

BLOCK = 5000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('keys_dir')
    p.add_argument('mps_pkl')
    p.add_argument('out_dir')
    p.add_argument('task', type=int)
    p.add_argument('n_tasks', type=int)
    args = p.parse_args()

    kx = np.load(Path(args.keys_dir) / 'keys_x.npy', mmap_mode='r')
    kz = np.load(Path(args.keys_dir) / 'keys_z.npy', mmap_mode='r')
    n = len(kx)
    start, stop = args.task * n // args.n_tasks, (args.task + 1) * n // args.n_tasks
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    with open(args.mps_pkl, 'rb') as f:
        mps = pickle.load(f)
    sites = mps_site_arrays(mps)

    t0, done = time.time(), 0
    for b0 in range(start, stop, BLOCK):
        b1 = min(b0 + BLOCK, stop)
        dest = out / f'vals_{b0:011d}_{b1:011d}.npy'
        if dest.exists():
            continue
        vals = batch_expectations(sites, np.asarray(kx[b0:b1]), np.asarray(kz[b0:b1]))
        tmp = dest.with_suffix(f'.tmp{os.getpid()}.npy')
        np.save(tmp, vals)
        os.replace(tmp, dest)
        done += b1 - b0
        print(f"task {args.task}: [{b0},{b1}) done, {done / (time.time() - t0):.1f} products/s", flush=True)
    print(f"task {args.task}: finished range [{start},{stop}) in {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
