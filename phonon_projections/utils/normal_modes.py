import numpy as np
from mlcalcdriver import Posinp
from mlcalcdriver.calculators import SchnetPackCalculator
from mlcalcdriver.workflows import Phonon
import h5py


def get_normal_modes(posinp, model, device="cpu"):
    calculator = SchnetPackCalculator(model, device=device)
    posinp = Posinp.from_file(posinp)
    ph = Phonon(posinp=posinp, calculator=calculator)
    ph.run()
    return ph.energies, ph.normal_modes
