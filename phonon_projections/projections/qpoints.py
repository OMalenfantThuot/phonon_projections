import numpy as np

def generate_qpoints_grid(X, Y, geometry="orthorhombic"):

    qpoints = []
    if geometry == "orthorhombic":
        xvect = np.array([1./X, 1./(4*Y), 0])
        yvect = np.array([0, 1./ Y, 0])
        basevect = np.array([0, 1./(2*Y),0])
        init = np.array([0, 0, 0])
        for i in range(X):
            for j in range(Y):
                qpoints.append(center(init + i * xvect + j * yvect))
                qpoints.append(center(init + i * xvect + j * yvect + basevect))

    elif geometry == "hexagonal":
        for i in range(X):
            for j in range(Y):
                qp = [i / X, j / Y, 0]
                qpoints.append(center(qp))
    return qpoints

def center(qp):
    idx = np.where(qp > 0.5)[0]
    qp[idx] = -1 + qp[idx]
    return qp
