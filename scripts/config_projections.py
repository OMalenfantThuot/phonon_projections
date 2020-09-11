from mlcalcdriver import Posinp
from mlcalcdriver.calculators import SchnetPackCalculator
from mlcalcdriver.workflows import Phonon
import numpy as np
import os.path
import h5py
import matplotlib.pyplot as plt

model = '/home/msadikov/projects/rrg-cotemich-ac/for_kevin/N_graphene_model'


def get_overlap_with_gamma_modes(posinp):
    calculator = SchnetPackCalculator(model)
    posinp = Posinp.from_file(posinp)
    ph = Phonon(posinp = posinp, calculator=calculator)
    ph.run()

    final_modes = np.zeros((96,96))
    for i in range(96):
        init = ph.normal_modes[:,i]
        final = np.empty_like(init)
        for j in range(96):
            if j%3 == 0:
                final[j] = init[j]
            elif j%3 == 1:
                final[j] = init[j+1]
            elif j%3 ==2 :
                final[j] = -1.0 * init[j-1]
        final_modes[:,i] = final
    energies = ph.energies


    g = h5py.File('/home/msadikov/projects/rrg-cotemich-ac/msadikov/programmes/phonon_projections/workdir/gamma_modes.h5', 'r')
    gam_mode_1 = g['first_mode']
    gam_mode_2 = g['second_mode']
    gam_energy_1 = 1483.5362685203481
    gam_energy_2 = 1483.5362685271216

    sum1=0
    sum2=0


    max1 = 0
    max2 = 0

    val_arr_1 = np.zeros(96)
    val_arr_2 = np.zeros(96)

    
    
    for i in range (96):
        defect_vec = final_modes[:,i]
        val1 = np.absolute(np.dot(defect_vec, gam_mode_1)).real**2
        val2 = np.absolute(np.dot(defect_vec, gam_mode_2)).real**2

        val_arr_1[i] = val1
        val_arr_2[i] = val2
        if val1>max1:
            max1=val1
            mode_dom_1 = i
        if val2>max2:
            max2 = val2
            mode_dom_2 = i


    val = val_arr_1 + val_arr_2



    #return mode_dom_1, mode_dom_2, max1, max2, energies[mode_dom_1], energies[mode_dom_2]
    return val, energies

pristine_file = '/home/msadikov/projects/rrg-cotemich-ac/msadikov/programmes/scripts_raman/databases/graphene_data/32_atoms/positions/32at_pristine.xyz'

#defect_file = '/home/msadikov/projects/rrg-cotemich-ac/msadikov/programmes/scripts_raman/databases/graphene_data/32_atoms_maria/2_def/32at_2d_000_(x1).xyz'


#val, energies = get_overlap_with_gamma_modes (pristine_file)
#plt.figure(figsize=(7,5))
#plt.xlabel('Energy')
#plt.ylabel('Projection on highest gamma modes')


#indices = np.argsort(energies)
#val_sorted = val[indices]

#energies_sorted = energies[indices]
#plt.plot(energies_sorted, val_sorted, linewidth=1)
#plt.scatter(energies, val, s=1, color='b')






maxrep = 7
n_atoms = 32
configs = 87
for a in range(configs):
    for b in np.linspace(1,maxrep,maxrep):
        b=int(b)
        defect_file = '/home/msadikov/projects/rrg-cotemich-ac/msadikov/programmes/scripts_raman/databases/graphene_data/32_atoms_maria/3_def/32at_3d_{:03}_(x{}).xyz'.format(a,b)
        if os.path.isfile(defect_file):
            val, energies = get_overlap_with_gamma_modes(defect_file)
            indices = np.argsort(energies)
            val_sorted = val[indices]
            energies_sorted = energies[indices]
            plt.figure(figsize=(7,5))
            plt.xlabel('Energy')
            plt.ylabel('Projection')
            plt.plot(energies_sorted, val_sorted)
            plt.savefig('../workdir/projection_plots/32at_3d_plots/config_{:03}_plot.png'.format(a)) 
