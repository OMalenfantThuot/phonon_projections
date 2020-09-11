import numpy as np

def project_mode(mode1, mode2):
    return np.absolute(np.vdot(mode1, mode2)).real **2
