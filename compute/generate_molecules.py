import sys
import json
import hashlib
from pathlib import Path

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'molecules'
BASIS = 'sto-3g'
MULTIPLICITY = 1
CHARGE = 0

# Experimental equilibrium geometries (Angstrom) as tabulated by NIST CCCBDB
# (https://cccbdb.nist.gov), Cartesian coordinates as listed there.
# H6 is a linear chain with 1.0 Angstrom spacing.
GEOMETRIES = {
    'LiH': {
        'geometry': [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.595))],
        'source': 'CCCBDB casno=7580678, r_e 7Li1H (NIST Diatomic Spectral Database)',
    },
    'BeH2': {
        'geometry': [('Be', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.3264)), ('H', (0.0, 0.0, -1.3264))],
        'source': 'CCCBDB casno=7787522; Shayesteh, Tereszchuk, Bernath, Colin, J. Chem. Phys. 118, 3622 (2003)',
    },
    'H2O': {
        'geometry': [('O', (0.0, 0.0, 0.1173)), ('H', (0.0, 0.7572, -0.4692)), ('H', (0.0, -0.7572, -0.4692))],
        'source': 'CCCBDB casno=7732185; Hoy and Bunker (1979)',
    },
    'NH3': {
        'geometry': [('N', (0.0, 0.0, 0.0)), ('H', (0.0, -0.9377, -0.3816)),
                     ('H', (0.8121, 0.4689, -0.3816)), ('H', (-0.8121, 0.4689, -0.3816))],
        'source': 'CCCBDB casno=7664417; Herzberg (1966)',
    },
    'CH4': {
        'geometry': [('C', (0.0, 0.0, 0.0)), ('H', (0.6276, 0.6276, 0.6276)), ('H', (0.6276, -0.6276, -0.6276)),
                     ('H', (-0.6276, 0.6276, -0.6276)), ('H', (-0.6276, -0.6276, 0.6276))],
        'source': 'CCCBDB casno=74828; Hirota (1979)',
    },
    'H6': {
        'geometry': [('H', (0.0, 0.0, 1.0 * i)) for i in range(6)],
        'source': 'Linear chain, 1.0 Angstrom spacing',
    },
}


def main():
    names = sys.argv[1:] or list(GEOMETRIES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in names:
        spec = GEOMETRIES[name]
        mol = MolecularData(spec['geometry'], BASIS, MULTIPLICITY, CHARGE, description=name,
                            filename=str(OUTPUT_DIR / name))
        mol = run_pyscf(mol, run_scf=True, run_fci=True)
        mol.save()
        path = Path(mol.filename + '.hdf5')
        manifest[name] = {
            'file': path.name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'geometry_angstrom': spec['geometry'],
            'source': spec['source'],
            'basis': BASIS, 'multiplicity': MULTIPLICITY, 'charge': CHARGE,
            'hf_energy': float(mol.hf_energy), 'fci_energy': float(mol.fci_energy),
            'n_qubits': 2 * mol.n_orbitals, 'n_electrons': mol.n_electrons,
        }
        print(f"{name}: {path.name}  HF {mol.hf_energy:.10f}  FCI {mol.fci_energy:.10f}")
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
