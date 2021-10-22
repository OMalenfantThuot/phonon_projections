from phonon_projections.utils import get_normal_modes
from phonon_projections.projections import project_mode
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt


def main(args):
    with h5py.File(args.primitive, "r") as f:
        energies = f["energies"][:]
        modes = f["modes"][:]

    pos_list = [pos for pos in os.listdir(args.pos_dir) if pos.endswith(".xyz")]
    calculator = SchnetPackCalculator(model)

    for i, pos in enumerate(pos_list):
        energies, modes = get_normal_modes(
            os.path.join(args.pos_dir, pos),
            args.model,
            device=args.device,
            rotate=args.rotate,
        )

        ndim = len(energies)

        for j in range(ndim):
            defect_mode = modes[:, j]

    sum1, sum2 = 0, 0
    max1, max2 = 0, 0
    val_arr_1 = np.zeros(96)
    val_arr_2 = np.zeros(96)

    for i in range(96):
        defect_vec = final_modes[:, i]
        val1 = np.absolute(np.dot(defect_vec, gam_mode_1)).real ** 2
        val2 = np.absolute(np.dot(defect_vec, gam_mode_2)).real ** 2

        val_arr_1[i] = val1
        val_arr_2[i] = val2
        if val1 > max1:
            max1 = val1
            mode_dom_1 = i
        if val2 > max2:
            max2 = val2
            mode_dom_2 = i
    val = val_arr_1 + val_arr_2

    return val, energies


pristine_file = "/home/msadikov/projects/rrg-cotemich-ac/msadikov/programmes/scripts_raman/databases/graphene_data/32_atoms/positions/32at_pristine.xyz"

maxrep = 7
n_atoms = 32
configs = 87
for a in range(configs):
    for b in np.linspace(1, maxrep, maxrep):
        b = int(b)
        defect_file = "/home/msadikov/projects/rrg-cotemich-ac/msadikov/programmes/scripts_raman/databases/graphene_data/32_atoms_maria/3_def/32at_3d_{:03}_(x{}).xyz".format(
            a, b
        )
        if os.path.isfile(defect_file):
            val, energies = get_overlap_with_gamma_modes(defect_file)
            indices = np.argsort(energies)
            val_sorted = val[indices]
            energies_sorted = energies[indices]
            plt.figure(figsize=(7, 5))
            plt.xlabel("Energy")
            plt.ylabel("Projection")
            plt.plot(energies_sorted, val_sorted)
            plt.savefig(
                "../workdir/projection_plots/32at_3d_plots/config_{:03}_plot.png".format(
                    a
                )
            )


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultHelpFormatter
    )
    parser.add_argument(
        "pos_dir", help="Directory containing the configurations to study."
    )
    parser.add_argument(
        "model", help="Path to the model used to generate the normal modes."
    )
    parser.add_argument(
        "primitive", help="Path to the h5 file containing the primitive cell modes."
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Add to rotate the modes from the xz plan to the xy plan.",
    )
    parser.add_argument("--device", default="cpu", help="Either 'cuda' or 'cpu'.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
