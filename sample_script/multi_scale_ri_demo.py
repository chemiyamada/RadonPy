#!/usr/bin/env python3

#  Copyright (c) 2023. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

"""Demonstration of refractive index calculations at multiple scales."""

__version__ = '0.1.0'

import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from radonpy.core import utils, poly, calc


def compute_ri(monomer_smi: str, scales: list[tuple[int, int]], density: float = 1.0) -> pd.DataFrame:
    """Generate cells and compute refractive index for each condition.

    Parameters
    ----------
    monomer_smi : str
        SMILES with two connection points (``*``) representing the monomer.
    scales : list[tuple[int, int]]
        List of ``(degree, molecule_count)`` pairs.
    density : float, optional
        Target density used for initial cell generation, by default ``1.0`` g/cm\ :sup:`3`.

    Returns
    -------
    pandas.DataFrame
        Results containing degree, molecule count, density and refractive index.
    """
    monomer = utils.mol_from_smiles(monomer_smi)
    mr_unit = Descriptors.MolMR(Chem.RemoveHs(monomer))
    mw_unit = Descriptors.MolWt(monomer)

    results = []
    for degree, count in scales:
        polymer = poly.polymerize_mols(monomer, n=degree)
        cell = poly.amorphous_cell(polymer, n=count, density=density)
        rho = calc.mol_density(cell)
        q = (mr_unit * degree) * rho / (mw_unit * degree)
        ri = np.sqrt((1 + 2 * q) / (1 - q))
        results.append({
            'degree': degree,
            'molecule_count': count,
            'density': rho,
            'refractive_index': ri
        })

    return pd.DataFrame(results)


def parse_scales(values: list[str]) -> list[tuple[int, int]]:
    """Parse ``degree,count`` pairs from command line."""
    scales = []
    for val in values:
        deg_str, cnt_str = val.split(',')
        scales.append((int(deg_str), int(cnt_str)))
    return scales


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refractive index demo')
    parser.add_argument('smiles', help='Monomer SMILES (use "*" for connection points)')
    parser.add_argument('--scale', nargs='+', required=True,
                        help='Pairs of "degree,count" e.g. 10,4 20,2')
    parser.add_argument('--density', type=float, default=1.0,
                        help='Target density for initial cell generation [g/cm^3]')
    args = parser.parse_args()

    scales = parse_scales(args.scale)
    df = compute_ri(args.smiles, scales, density=args.density)
    print(df.to_string(index=False))
