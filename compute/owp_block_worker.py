import os
import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pauli_sweep import mps_site_arrays, batch_expectations

# Same kernel and block naming as owp_exp_worker.py, but takes an explicit list of blocks
# (lines "<keys_dir> <b0> <b1>"); this worker handles lines task, task + n_tasks, ...


def main():
    p = argparse.ArgumentParser()
    p.add_argument('block_list')
    p.add_argument('mps_pkl')
    p.add_argument('task', type=int)
    p.add_argument('n_tasks', type=int)
    args = p.parse_args()

    lines = Path(args.block_list).read_text().split('\n')
    mine = [l.split() for l in lines if l.strip()][args.task::args.n_tasks]
    with open(args.mps_pkl, 'rb') as f:
        sites = mps_site_arrays(pickle.load(f))
    keys = {}
    t0, done = time.time(), 0
    for kd, b0, b1 in mine:
        b0, b1 = int(b0), int(b1)
        dest = Path(kd) / 'vals' / f'vals_{b0:011d}_{b1:011d}.npy'
        if dest.exists():
            continue
        if kd not in keys:
            keys[kd] = (np.load(Path(kd) / 'keys_x.npy', mmap_mode='r'), np.load(Path(kd) / 'keys_z.npy', mmap_mode='r'))
        kx, kz = keys[kd]
        vals = batch_expectations(sites, np.asarray(kx[b0:b1]), np.asarray(kz[b0:b1]))
        tmp = dest.with_suffix(f'.tmp{os.uname().nodename}_{os.getpid()}.npy')
        np.save(tmp, vals)
        os.replace(tmp, dest)
        done += b1 - b0
        print(f"task {args.task}: {kd} [{b0},{b1}) done, {done / (time.time() - t0):.1f} products/s", flush=True)
    print(f"task {args.task}: finished {len(mine)} blocks in {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
