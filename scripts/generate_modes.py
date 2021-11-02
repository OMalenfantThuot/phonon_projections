#! /usr/bin/env python

import argparse
from phonon_projections.utils import get_normal_modes
import h5py


def main(args):
    energies, modes = get_normal_modes(
        args.posinp, args.model, device=args.device, rotate=args.rotate
    )
    filename = args.name if args.name.endswith(".h5") else args.name + ".h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("modes", data=modes)
        f.create_dataset("energies", data=energies)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model", help="Path to the model used to generate the normal modes."
    )
    parser.add_argument(
        "posinp", help="Path to position file containing the equilibrium positions."
    )
    parser.add_argument("--device", default="cpu", help="Either 'cuda' or 'cpu'.")
    parser.add_argument(
        "--name",
        help="Name of the h5 file containing the modes.",
        type=str,
        default="normal_modes",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
