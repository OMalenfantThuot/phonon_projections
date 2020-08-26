import numpy as np
import shutil


def stackModesForSmallCell(ddb, sorted=False):
    all_eigdis = []
    all_eigs = []
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
        all_eigs.append(bands.phfreqs * 8065.5)
        all_eigdis.append(bands.dyn_mat_eigenvect[0])

    all_eigs = np.array(all_eigs).flatten()
    all_eigdis = np.array(all_eigdis)

    # this is the size difference between the two cells
    sizes = [4, 2]
    norm = 1 / np.sqrt(2 * np.prod(sizes))
    new_modes = []
    lattice = ddb.structure.lattice.matrix
    a1 = lattice[0]
    a2 = 2 * lattice[1] - lattice[0]

    for i, qpt in enumerate(ddb.qpoints):
        for j, mode in enumerate(all_eigdis[i]):
            for m in range(1, sizes[1] + 1):
                for n in range(1, sizes[0] + 1):
                    # check for complex phase
                    rvec = a1 * n + a2 * m
                    if n == 1 and m == 1:
                        first_piece = mode * np.exp(1j * np.dot(qpt.cart_coords, rvec))
                        second_piece = mode * np.exp(
                            1j * np.dot(qpt.cart_coords, rvec + lattice[1])
                        )
                        large_mode = np.hstack((first_piece, second_piece))
                    else:
                        first_piece = mode * np.exp(1j * np.dot(qpt.cart_coords, rvec))
                        second_piece = mode * np.exp(
                            1j * np.dot(qpt.cart_coords, rvec + lattice[1])
                        )
                        large_mode = np.hstack(
                            (large_mode, np.hstack((first_piece, second_piece)))
                        )
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
