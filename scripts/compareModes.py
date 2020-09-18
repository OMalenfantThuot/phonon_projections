#!/usr/bin/env python

from abipy.dfpt.ddb import DdbFile as DDB
import numpy as np
import shutil
import argparse
import matplotlib.pyplot as plt
import h5py
from phonon_projections.projections import stackModesForSmallCell, project_mode


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    run_type_subparser = parser.add_subparsers(
        dest="run_type", help="Chooses the type of run."
    )

    # Displacements analysis
    displacements_parser = run_type_subparser.add_parser(
        "disp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Displacements analysis.",
    )
    displacements_parser.add_argument(
        "-s",
        "--small",
        type=str,
        help="DDB file to open with the primitive unit cell modes.",
    )
    displacements_parser.add_argument(
        "-f", "--filename", type=str, help="h5 file containing the displacements."
    )
    displacements_parser.add_argument(
        "--size",
        nargs=2,
        help="Size of the supercell in number of primitive cells [X, Y].",
        type=int,
    )
    displacements_parser.add_argument(
        "-g",
        "--geo",
        default="o",
        help="Geometry of the supercell, either orthorhombic or hexagonal.",
        choices=["o", "h"],
    )

    # Modes projection to primitive cell
    mode_parser = run_type_subparser.add_parser(
        "modes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Modes projection.",
    )
    mode_parser.add_argument(
        "--small",
        type=str,
        help="DDB file to open with the primitive unit cell modes.",
    )
    mode_parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="h5 file containing the modes and energies of the supercell.",
    )
    mode_parser.add_argument(
        "--size",
        nargs=2,
        help="Size of the supercell in number of primitive cells [X, Y].",
        type=int,
    )
    mode_parser.add_argument(
        "-g",
        "--geo",
        default="o",
        help="Geometry of the supercell, either orthorhombic or hexagonal.",
        choices=["o", "h"],
    )
    return parser


def main(args):
    if shutil.which("anaddb") is None:
        print("You must have the ABINIT executables in your path ... Exiting.")
        exit(-1)

    small_ddb = DDB(args.small)
    seigs, svecs = stackModesForSmallCell(
        small_ddb, args.size, geometry=args.geo, sorted=False
    )
    natoms = 4 * np.prod(args.size) if args.geo == "o" else 2 * np.prod(args.size)

    if args.run_type == "disp":
        bvecs = h5py.File(args.filename, "r")["displacements"][:].reshape(
            -1, 3 * natoms
        )
        sums = np.zeros(svecs.shape[0])
        for n in range(bvecs.shape[0]):
            bvec = bvecs[n]
            for j, svec in enumerate(svecs):
                val = project_mode(bvec, svec)
                sums[j] += val
        labels = []
        for j, svec in enumerate(svecs):
            labels.append(str(small_ddb.qpoints[j // 6].frac_coords) + "-" + str(j % 6))
            print(
                "---seig:",
                j % 6,
                "qpoint:",
                small_ddb.qpoints[j // 6].frac_coords,
                "overlap sum:",
                sums[j],
                "frequency [cm^-1]:",
                seigs[j],
            )
        E = np.linspace(np.min(seigs), np.max(seigs), 4096)
        func = np.zeros_like(E)
        for i, eig in enumerate(seigs):
            func += sums[i] * np.exp(-((E - eig) ** 2) / 100)
        plt.plot(E, func)
        plt.tight_layout()
        plt.savefig("figure.png")

    elif args.run_type == "modes":
        with h5py.File(args.filename, "r") as f:
            bvecs = f["modes"][:]
            beigs = f["energies"][:]
        nmodes = len(beigs)
        maxs, sums = np.zeros(nmodes), np.zeros(nmodes)
        vec_ids = np.zeros(nmodes).astype(int)
        for n in range(nmodes):
            bvec = bvecs[:, n]
            for j, svec in enumerate(svecs):
                val = project_mode(bvec, svec)
                sums[n] += val
                if val > maxs[n]:
                    maxs[n], vec_ids[n] = val, j

        for n in range(nmodes):
            if maxs[n] > 0.9:
                print(
                    "big frequency",
                    beigs[n],
                    "overlap:",
                    maxs[n],
                    "small frequency [cm^-1]:",
                    seigs[vec_ids[n]],
                    "sum:",
                    sums[n],
                )
    else:
        raise ValueError("run_type not recognized.")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
