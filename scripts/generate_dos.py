#! /usr/bin/env python

from schnetpack.utils import load_model
from mlcalcdriver import Posinp
from mlcalcdriver.interfaces import posinp_to_ase_atoms
from schnetpack.environment import AseEnvironmentProvider
from schnetpack.interfaces import SpkCalculator
from ase.phonons import Phonons
import argparse
import h5py
from phonon_projections.dos import Dos


def main(args):
    atoms = posinp_to_ase_atoms(Posinp.from_file(args.posinp))
    model = load_model(args.model, map_location=args.device)
    
    # Bugfix to make older models work with PyTorch 1.6
    # Hopefully temporary
    for mod in model.modules():
        if not hasattr(mod, "_non_persistent_buffers_set"):
            mod._non_persistent_buffers_set = set()

    cutoff = float(
        model.state_dict()["representation.interactions.0.cutoff_network.cutoff"]
    )
    calculator = SpkCalculator(
        model,
        device=args.device,
        energy="energy",
        forces="forces",
        environment_provider=AseEnvironmentProvider(cutoff),
    )
    #supercell = tuple(args.supercell)
    ph = Phonons(atoms, calculator, supercell=args.supercell, delta=0.02)
    ph.run()
    ph.read(acoustic=True)
    ph.clean()
    dos = ph.get_dos(kpts=args.qpoints).sample_grid(npts=args.npts, width=args.width)
    
    if args.output:
        with h5py.File("dos.h5", "w") as f:
            f.create_dataset("weights", data=dos.weights[0])
            f.create_dataset("energies", data=dos.energy)

    return Dos(dos.energy, dos.weights[0])


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model", help="Path to the model used to generate the phonon dos."
    )
    parser.add_argument(
        "posinp", help="Path to position file containing the equilibrium positions."
    )
    parser.add_argument("--device", default="cpu", help="Either 'cuda' or 'cpu'.")
    parser.add_argument(
        "--supercell", help="Size of the supercell.", nargs=3, default=[6, 6, 6], type=int,
    )
    parser.add_argument(
        "--qpoints",
        help="Qpoints grid for the dos estimation.",
        nargs=3,
        type=int,
        default=[30, 30, 30],
    )
    parser.add_argument(
        "--npts", help="Resolution in energy for the dos.", default=1000
    )
    parser.add_argument(
        "--width", help="Gaussian smearing to build the dos.", default=0.004
    )
    parser.add_argument("--output", default=False, action="store_true", help="If used, writes the dos to disk.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.supercell  = tuple(args.supercell)
    dos = main(args)
