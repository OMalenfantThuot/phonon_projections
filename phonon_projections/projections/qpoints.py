def generate_qpoints_grid(X, Y, geometry="orthorhombic"):

    qpoints = []
    if geometry == "orthorhombic":
        for i in range(X):
            for j in range(2 * Y):
                qp = [i / X, j / (2 * Y), 0]
                qpoints.append(qp)
    elif geometry == "hexagonal":
        for i in range(X):
            for j in range(Y):
                qp = [i / X, j / Y, 0]
                qpoints.append(qp)
    return qpoints
