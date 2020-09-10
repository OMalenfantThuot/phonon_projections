import numpy as np
import shutil


def stackModesForSmallCell(ddb, sizes, geometry="orthorhombic", sorted=False):

    all_eigdis, all_eigs = [], []
    assert len(sizes) == 2, "Sizes must be of length 2."

    for qpt in ddb.qpoints:
        print("Working on qpt:", qpt)
        shutil.rmtree("./tmp", ignore_errors=True)
        bands = ddb.anaget_phmodes_at_qpoint(
            qpt,
            asr=1,
            chneut=1,
            dipdip=1,
            workdir="./tmp",
            verbose=0,
            anaddb_kwargs={"eivec": 1},
        )
        all_eigs.append(bands.phfreqs)
        all_eigdis.append(bands.dyn_mat_eigenvect[0])

    all_eigs = np.array(all_eigs).flatten() * 8065.5
    all_eigdis = np.array(all_eigdis)

    lattice = ddb.structure.lattice.matrix
    new_modes = []
    if geometry in ["o", "orthorhombic"]:
        norm = 1 / np.sqrt(2 * np.prod(sizes))
        a1, a2 = lattice[0], 2 * lattice[1] - lattice[0]
    elif geometry in ["h", "hexagonal"]:
        norm = 1 / np.sqrt(np.prod(sizes))
        a1, a2 = lattice[0], lattice[1]

    for i, qpt in enumerate(ddb.qpoints):
        for j, mode in enumerate(all_eigdis[i]):
            if geometry in ["o", "orthorhombic"]:
                for m in range(sizes[0]):
                    for n in range(sizes[1]):
                        # check for complex phase
                        rvec = a1 * m + a2 * n
                        first_piece = mode * np.exp(1j * np.dot(qpt.cart_coords, rvec))
                        second_piece = mode * np.exp(
                            1j * np.dot(qpt.cart_coords, rvec + lattice[1])
                        )
                        if n == 0 and m == 0:
                            large_mode = np.hstack((first_piece, second_piece))
                        else:
                            large_mode = np.hstack(
                                (large_mode, np.hstack((first_piece, second_piece)))
                            )
                new_modes.append(norm * large_mode)

            elif geometry in ["h", "hexagonal"]:
                for m in range(sizes[0]):
                    for n in range(sizes[1]):
                        rvec = a1 * m + a2 * n
                        piece = mode * np.exp(1j * np.dot(qpt.cart_coords, rvec))
                        if n == 0 and m == 0:
                            large_mode = piece.copy()
                        else:
                            large_mode = np.hstack((large_mode, piece))
                new_modes.append(norm * large_mode)

    new_modes = np.array(new_modes)
    if not sorted:
        return all_eigs, new_modes
    else:
        sorted_indices = np.argsort(all_eigs)
        return all_eigs[sorted_indices], new_modes[sorted_indices]


def getModesAtGamma(ddb):
    shutil.rmtree("./tmp", ignore_errors=True)
    bands = ddb.anaget_phmodes_at_qpoint(
        ddb.qpoints[0],
        asr=1,
        chneut=1,
        dipdip=1,
        workdir="./tmp",
        verbose=0,
        anaddb_kwargs={"eivec": 1},
    )
    eigdis = bands.dyn_mat_eigenvect[0]
    return bands.phfreqs[0] * 8065.5, eigdis
