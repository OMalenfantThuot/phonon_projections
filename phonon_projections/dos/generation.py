from schnetpack.utils import load_model
from mlcalcdriver import Posinp
from mlcalcdriver.interfaces import posinp_to_ase_atoms
from schnetpack.environment import AseEnvironmentProvider
from schnetpack.interfaces import SpkCalculator
from phonon_projections.dos import Dos
from ase.phonons import Phonons
import torch
import h5py


def get_dos(
    model,
    posinp,
    device="cpu",
    supercell=(6, 6, 6),
    qpoints=[30, 30, 30],
    npts=1000,
    width=0.004,
):
    if isinstance(posinp, str):
        atoms = posinp_to_ase_atoms(Posinp.from_file(posinp))
    elif isinstance(posinp, Posinp):
        atoms = posinp_to_ase_atoms(posinp)
    else:
        raise ValueError("The posinp variable is not recognized.")

    if isinstance(model, str):
        model = load_model(model, map_location=device)
    elif isinstance(model, torch.nn.Module):
        pass
    else:
        raise ValueError("The model variable is not recognized.")

    # Bugfix to make older models work with PyTorch 1.6
    # Hopefully temporary
    for mod in model.modules():
        if not hasattr(mod, "_non_persistent_buffers_set"):
            mod._non_persistent_buffers_set = set()

    assert len(supercell) == 3, "Supercell should be a length 3 object."
    assert len(qpoints) == 3, "Qpoints should be a length 3 object."
    supercell = tuple(supercell)

    cutoff = float(
        model.state_dict()["representation.interactions.0.cutoff_network.cutoff"]
    )
    calculator = SpkCalculator(
        model,
        device=device,
        energy="energy",
        forces="forces",
        environment_provider=AseEnvironmentProvider(cutoff),
    )
    ph = Phonons(atoms, calculator, supercell=supercell, delta=0.02)
    ph.run()
    ph.read(acoustic=True)
    dos = ph.get_dos(kpts=qpoints).sample_grid(npts=npts, width=width)
    ph.clean()
    return Dos(dos.energy * 8065.6, dos.weights[0])
