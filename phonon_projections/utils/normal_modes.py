import numpy as np
from mlcalcdriver import Posinp
from mlcalcdriver.calculators import SchnetPackCalculator
from mlcalcdriver.workflows import Phonon
from copy import copy


def get_normal_modes(posinp, model, device="cpu", rotate=False):
    calculator = SchnetPackCalculator(model, device=device)
    posinp = Posinp.from_file(posinp)
    ph = Phonon(posinp=posinp, calculator=calculator)
    ph.run()

    if rotate:
        ndim = 3 * len(posinp)
        final_modes = np.zeros((ndim, ndim))

        idx = np.arange(ndim)
        for i in range(ndim):
            init = ph.normal_modes[:, i]
            final = np.empty_like(init)
            final[np.where(idx % 3 == 0)[0]] = init[np.where(idx % 3 == 0)[0]]
            final[np.where(idx % 3 == 1)[0]] = init[np.where(idx % 3 == 2)[0]]
            final[np.where(idx % 3 == 2)[0]] = -1.0 * init[np.where(idx % 3 == 1)[0]]
            final_modes[:, i] = final
        return ph.energies, final_modes

    else:
        return ph.energies, ph.normal_modes


def convert_modes_to_pixels(posinp, modes):
    posinp = Posinp.from_file(posinp)
    init_image = posinp.to_pixels()
    i1, i2, i3 = np.where(init_image > 0.5)
    images = []

    for i in range(3 * len(posinp)):
        mode = modes[:, i].reshape(-1, 3)
        image = copy(init_image)
        new = np.zeros_like(image)
        image = np.stack([image, new, new, new], axis=-1)
        image[i1, i2, i3, 1:] = mode
        images.append(image)

    return np.array(images)
