#! /usr/bin/env python

from phonon_projections.utils import get_orthonormal_basis
import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to the model used to guess normal modes.")
    parser.add_argument(
        "posinp", help="Path to the position file containing equilibrium positions."
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    _ = get_orthonormal_basis(model=args.model, posinp=args.posinp, write=True)
