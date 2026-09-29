import sys, time, numpy as np
from pathlib import Path
out = Path(sys.argv[1]); samples = [int(s) for s in sys.argv[2].split(',')]; dest = Path(sys.argv[3])
exclude_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
def uniq(x, z):
    o = np.lexsort((z, x)); x, z = x[o], z[o]
    k = np.ones(len(x), bool); k[1:] = (x[1:] != x[:-1]) | (z[1:] != z[:-1]); return x[k], z[k]
ux = uz = None
for s in samples:
    t = time.time(); p = np.load(out / f'products_seed0_sample{s}.npz')
    ux, uz = (p['x'], p['z']) if ux is None else uniq(np.concatenate([ux, p['x']]), np.concatenate([uz, p['z']]))
    print(f"after sample {s}: {len(ux):,} ({time.time()-t:.0f}s)", flush=True)
if exclude_dir is not None:  # drop keys already assigned to an earlier batch
    ex, ez = np.load(exclude_dir / 'keys_x.npy'), np.load(exclude_dir / 'keys_z.npy')
    ax, az = np.concatenate([ux, ex]), np.concatenate([uz, ez]); o = np.lexsort((az, ax)); ax, az = ax[o], az[o]
    dup = np.zeros(len(ax), bool); d = (ax[1:] == ax[:-1]) & (az[1:] == az[:-1]); dup[1:] |= d; dup[:-1] |= d
    newm = np.zeros(len(ax), bool); newm[o < len(ux)] = True   # positions coming from ux
    keep = newm & ~dup; ux, uz = ax[keep], az[keep]
    print(f"new relative to {exclude_dir.name}: {len(ux):,}", flush=True)
dest.mkdir(parents=True, exist_ok=True)
np.save(dest / 'keys_x.npy', ux); np.save(dest / 'keys_z.npy', uz)
(dest / 'SAMPLES').write_text(','.join(map(str, samples)) + '\n')
print(f"saved {len(ux):,} keys to {dest}", flush=True)
