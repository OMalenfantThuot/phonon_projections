#!/usr/bin/env python

from pymatgen.core.periodic_table import Element
from abipy.dfpt.ddb import DdbFile as DDB
import numpy as np
import sys
import os
import shutil
import argparse
import matplotlib.pyplot as plt
import h5py

def parseArgs():
    parser = argparse.ArgumentParser(description='Project eigendisplacements of larger cell onto smaller cell.')
    parser.add_argument('-s', '--small', type=str, help='File to open with the smaller unit cell')
    parser.add_argument('-b', '--big', type=str, help='File to open with the bigger unit cell')

    return parser.parse_args()

def getModesAtGamma(ddb):
    
    shutil.rmtree('./tmp', ignore_errors=True)
    bands = ddb.anaget_phmodes_at_qpoint(ddb.qpoints[0],
                                     asr=1,
                                     chneut=1,
                                     dipdip=1,
                                     workdir='./tmp',
                                     verbose=0,
                                     anaddb_kwargs={'eivec': 1})
     
    eigdis = bands.dyn_mat_eigenvect[0]

    return bands.phfreqs[0] * 8065.5, eigdis

def stackModesForSmallCell(ddb, sorted=False):
    all_eigdis = []
    all_eigs = []
    for qpt in ddb.qpoints:
        print('Working on qpt:', qpt)
        shutil.rmtree('./tmp', ignore_errors=True)
        bands = ddb.anaget_phmodes_at_qpoint(qpt,
                                     asr=1,
                                     chneut=1,
                                     dipdip=1,
                                     workdir='./tmp',
                                     verbose=0,
                                     anaddb_kwargs={'eivec': 1})
        all_eigs.append(bands.phfreqs * 8065.5)
        all_eigdis.append(bands.dyn_mat_eigenvect[0])

    all_eigs = np.array(all_eigs).flatten()
    all_eigdis = np.array(all_eigdis)
    
    # this is the size difference between the two cells
    sizes = [4, 2]
    norm = 1 / np.sqrt(np.prod(sizes))
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
                        second_piece = mode * np.exp(1j * np.dot(qpt.cart_coords, rvec + lattice[1]))
                        large_mode = np.hstack((first_piece, second_piece))
                    else:
                        first_piece = mode * np.exp(1j * np.dot(qpt.cart_coords, rvec))
                        second_piece = mode * np.exp(1j * np.dot(qpt.cart_coords, rvec + lattice[1]))
                        large_mode = np.hstack((large_mode, np.hstack((first_piece, second_piece))))
            new_modes.append(norm * large_mode)
    
    new_modes = np.array(new_modes)
    if not sorted:
        return all_eigs, new_modes
    else:
        sorted_indices = np.argsort(all_eigs)
        return all_eigs[sorted_indices], new_modes[sorted_indices]

def readDDB(fname):
    return DDB(fname)


def main():
    if shutil.which('anaddb') is None:
        print('You must have the ABINIT executables in your path ... Exiting.')
        exit(-1)
    
    args = parseArgs()
    labels = ['b', 'r']
    small_ddb = readDDB(args.small)
    # big_ddb = readDDB(args.big)
    seigs, svecs  = stackModesForSmallCell(small_ddb, sorted=False)
    # beigs, bvecs = getModesAtGamma(big_ddb)
    bvecs = h5py.File('new.h5')['displacements'][:].reshape(1000, 96)
    sums = np.zeros(svecs.shape[0])
    for n in range(bvecs.shape[0]):
        bvec = bvecs[n]
        prods = []
        for j, svec in enumerate(svecs):
            val = np.absolute(np.vdot(bvec, svec)).real**2
            sums[j] += val
        prods.append(val)
    labels = []
    for j, svec in enumerate(svecs):
        labels.append(str(small_ddb.qpoints[j // 6].frac_coords) + '-' + str(j % 6))
        print('---seig:', j % 6, 'qpoint:', small_ddb.qpoints[j // 6].frac_coords, 'overlap sum:', sums[j], 'frequency [cm^-1]:', seigs[j]) 
    E = np.linspace(np.min(seigs), np.max(seigs), 4096)
    func = np.zeros_like(E)
    for i, eig in enumerate(seigs):
        func += sums[i] * np.exp(-(E - eig)**2 / 100)
    plt.plot(E, func)
    # plt.bar(range(len(sums)), sums)
    # plt.xticks(range(len(labels)), labels, rotation=90)
    plt.tight_layout()
    plt.show()
       # sorted_inds = np.flip(np.argsort(prods))
       # print('beig:', i, 'frequency [cm^-1]:', beigs[i])
       # for j in range(5):
       #    print('---seig:', sorted_inds[j] % 6, 'qpoint:', small_ddb.qpoints[sorted_inds[j] // 6].frac_coords, 'overlap:', prods[sorted_inds[j]], 'frequency [cm^-1]:', seigs[sorted_inds[j]])

if __name__ == '__main__':
    main()
