import numpy as np


def project_mode(mode1, mode2):
    p = np.vdot(mode1, mode2)
    return (p * p.conjugate()).real
