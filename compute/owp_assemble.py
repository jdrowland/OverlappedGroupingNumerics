import json
import re
import argparse
from pathlib import Path

import numpy as np


def load_blocks(vals_dir, threads=32):
    # completed blocks only; workers write vals_<b0>_<b1>.tmp<pid>.npy and rename atomically
    from concurrent.futures import ThreadPoolExecutor
    files = []
    for f in sorted(Path(vals_dir).iterdir()):
        m = re.fullmatch(r'vals_(\d{11})_(\d{11})\.npy', f.name)
        if m:
            files.append((int(m.group(1)), int(m.group(2)), f))
    with ThreadPoolExecutor(threads) as ex:
        yield from ex.map(lambda t: (t[0], t[1], np.load(t[2])), files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')                 # .../ogn_ties/out
    p.add_argument('--write-missing', default=None)
    args = p.parse_args()
    out = Path(args.out)

    nA = len(np.load(out / 'batchA/keys_x.npy', mmap_mode='r'))
    nB = len(np.load(out / 'batchB/keys_x.npy', mmap_mode='r'))
    vA, vB = np.full(nA, np.nan), np.full(nB, np.nan)

    for b0, b1, v in load_blocks(out / 'batchA/vals'):
        vA[b0:b1] = v

    seg = json.loads((out / 'pool/SEGMENTS.json').read_text())
    pool_to = []   # (pool_start, pool_stop, target array, src_start)
    for s in seg['segments']:
        pool_to.append((s['pool_start'], s['pool_stop'], vA if s['source'] == 'batchA' else vB, s['src_start']))
    for b0, b1, v in load_blocks(out / 'pool/vals'):
        for ps, pe, arr, ss in pool_to:
            lo, hi = max(b0, ps), min(b1, pe)
            if lo < hi:
                arr[ss + lo - ps: ss + hi - ps] = v[lo - b0: hi - b0]

    missA, missB = np.isnan(vA), np.isnan(vB)
    print(f"batchA: {nA:,} keys, {missA.sum():,} missing ({missA.mean():.2%})")
    print(f"batchB: {nB:,} keys, {missB.sum():,} missing ({missB.mean():.2%})")
    print(f"total missing: {missA.sum() + missB.sum():,} of {nA + nB:,}")

    if args.write_missing:
        d = Path(args.write_missing); d.mkdir(parents=True, exist_ok=True)
        ax, az = np.load(out / 'batchA/keys_x.npy', mmap_mode='r'), np.load(out / 'batchA/keys_z.npy', mmap_mode='r')
        bx, bz = np.load(out / 'batchB/keys_x.npy', mmap_mode='r'), np.load(out / 'batchB/keys_z.npy', mmap_mode='r')
        np.save(d / 'keys_x.npy', np.concatenate([ax[missA], bx[missB]]))
        np.save(d / 'keys_z.npy', np.concatenate([az[missA], bz[missB]]))
        print(f"wrote {missA.sum() + missB.sum():,} missing keys to {d}")
    elif missA.sum() + missB.sum() == 0:
        np.save(out / 'batchA/vals_all.npy', vA)
        np.save(out / 'batchB/vals_all.npy', vB)
        print("complete: wrote batchA/vals_all.npy and batchB/vals_all.npy")


if __name__ == '__main__':
    main()
