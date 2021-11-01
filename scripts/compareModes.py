#!/usr/bin/env python

from utils.vibrations import GrapheneDDB
import numpy as np
import shutil
import argparse
import matplotlib.pyplot as plt
import h5py
from phonon_projections.projections import project_mode


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "ddb", type=str, help="DDB file to open with the primitive unit cell modes.",
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
        "-f", "--dispfile", type=str, help="h5 file containing the displacements."
    )

    # Modes projection to primitive cell
    mode_parser = run_type_subparser.add_parser(
        "modes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Modes projection.",
    )
    mode_parser.add_argument(
        "-f",
        "--modesfile",
        type=str,
        help="h5 file containing the modes and energies of the supercell.",
    )
    mode_parser.add_argument(
        "--write",
        action="store_true",
        help="If used, the energy projection of the modes is written.",
    )
    return parser


def main(args):
    if shutil.which("anaddb") is None:
        print("You must have the ABINIT executables in your path ... Exiting.")
        exit(-1)

    gDDB = GrapheneDDB(args.ddb)
    seigs, svecs = [], []
    for i, qpoint in enumerate(gDDB.qpoints):
        for branch in range(6):
            v, e = gDDB.build_supercell_modes(
                qpoint.frac_coords, branch, energies=True, return_positions=False,
            )
            norm = np.linalg.norm(v)
            svecs.append(v / norm)
            seigs.append(e)

    seigs, svecs = np.array(seigs), np.array(svecs)
    natoms = 2 * i

    if args.run_type == "disp":
        bvecs = h5py.File(args.dispfile, "r")["displacements"][:].reshape(
            -1, 3 * natoms
        )
        sums = np.zeros(svecs.shape[0])
        for n in range(bvecs.shape[0]):
            bvec = bvecs[:, n]
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
        with h5py.File(args.modesfile, "r") as f:
            try:
                bvecs = f["modes"][:]
                beigs = f["energies"][:]
            except:
                bvecs = f["basis"][:]
                beigs = np.zeros(bvecs.shape[0])
        nmodes = len(beigs)
        maxs, sums, energies = np.zeros(nmodes), np.zeros(nmodes), np.zeros(nmodes)
        vec_ids = np.zeros(nmodes).astype(int)

        for n in range(nmodes):
            bvec = bvecs[:, n]
            for j, svec in enumerate(svecs):
                val = project_mode(bvec, svec)
                sums[n] += val
                energies[n] += val * seigs[j]
                if val > maxs[n]:
                    maxs[n], vec_ids[n] = val, j

        assert np.allclose(sums, np.ones(nmodes), atol=1.0e-6)

        if args.write:
            with h5py.File(args.modesfile, "r+") as f:
                f.create_dataset("projected_energies", data=energies)

    else:
        raise ValueError("run_type not recognized.")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
