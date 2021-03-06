from mlcalcdriver import Posinp
from mlcalcdriver.calculators import SchnetPackCalculator
from mlcalcdriver.workflows import Phonon
from phonon_projections.utils import gramschmidt
import h5py

def get_orthonormal_basis(modes=None, model=None, posinp=None, write=False):
    if modes:
        with h5py.File(modes, "r") as f:
            normal_modes = f["modes"][:]
            orthonormal_basis = gramschmidt(normal_modes)
    else:
        calculator = SchnetPackCalculator(model)
        posinp = Posinp.from_file(posinp)
        ph = Phonon(posinp=posinp, calculator=calculator)
        ph.run()

        normal_modes = ph.normal_modes.copy()
        orthonormal_basis = gramschmidt(normal_modes)
    if write:
        fx = h5py.File("orthonormal_basis.h5", "w")
        fx.create_dataset("basis", data=orthonormal_basis)
        fx.close()
    return orthonormal_basis
