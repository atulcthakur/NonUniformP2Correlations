#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy as sp
import numba
from tqdm import tqdm
from numba import jit, njit
from useful import read_lmp_xyz_faster, xyz_writer, make_whole, sort_traj, map_index, pickle_object, unpickle_object, save_arrays_to_file
from numba.typed import List
from numba.types import float64, int64
import pickle
import matplotlib.pyplot as plt


# In[ ]:


import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


# In[ ]:


@jit(nopython=True)
def direct_correlate(array1,array2,array3):
    assert len(array1)==len(array2)==len(array3),"Your code broke because of wrong inputs"
    #print("This code will calculate P2 correlation only")
    t_cor = len(array1)
    t_run = len(array1)
    c_t = np.zeros(t_cor)
    for time in range(t_cor):
        t_max =  t_run - time
        #c_t[time] = sum(array[0:t_max]*array[time:t_max+time]) / t_max
        c_t[time] = np.sum(((array1[0:t_max]*array1[time:t_max+time]) + (array2[0:t_max]*array2[time:t_max+time]) + (array3[0:t_max]*array3[time:t_max+time]))**2) / t_max
    return (3/2)*c_t-(1/2)


# In[ ]:





# *Steps*
# - Curate the trajectory. 
# - Select oxygens within z.
# - Make a timeseries with oxygens
# - Select those molecules. 
# - Compute O-H vectors. 
# - Do P2. 

# In[ ]:


def center_boxcenter(traj, boxcenter):
    """
    Will modify the co-ordinates so that the center is at boxcenter.
    Boxcenter list has to be computed externally.
    """
    boxcenter = np.array(boxcenter)
    out = {}
    for time in tqdm(traj):
        frame = traj[time].copy(deep=True)
        frame[['x', 'y', 'z']] = frame[['x', 'y', 'z']] - boxcenter
        out[time] = frame
    return out


# In[ ]:


def density_profile(traj, atom_traj=None):
    """
    Will compute density profile along z
    """
    all_z = []
    for time in tqdm(traj):
        frame = traj[time].copy(deep=True)
        if atom_traj is not None:
            frame = frame.loc[frame['atoms'] == atom_traj]
        frame_z = frame['z'].to_numpy()
        all_z.append(frame_z)
    dens, z_bins = np.histogram(all_z, bins=500)
    bins = (z_bins[:-1] + z_bins[1:])/2
    return dens, bins


# In[ ]:


def curate_traj(lmpxyz="file.xyz", nmol=1024):
    traj = {}
    lmpxyz_traj, natoms = read_lmp_xyz_faster(lmpxyz)
    for time in tqdm(lmpxyz_traj):
        frame = lmpxyz_traj[time].copy(deep=True)
        mols = np.repeat(np.arange(1024), 3)
        frame['mols'] = mols
        traj[time] = frame
    return traj


# In[ ]:


def select_layer_oxygens(traj, z_hi=5.7, z_lo=0):
    layer_traj = {}
    unique_mols = []
    for time in tqdm(traj):
        frame = traj[time].copy(deep=True)
        frame_o = frame.loc[frame['atoms'] == 'O']
        first_layer = frame_o.loc[(z_hi >= frame_o['z']) & (frame_o['z'] >  z_lo)]
        last_layer = frame_o.loc[(-z_lo > frame_o['z']) & (frame_o['z'] >= -z_hi)]
        total_layer = pd.concat([first_layer, last_layer]).reset_index(drop=True)
        layer_traj[time] = total_layer
        total_layer_mols = total_layer['mols'].to_numpy()
        unique_mols.append(total_layer_mols)
    unique_mols_allframes = np.unique(np.concatenate(unique_mols).flatten())
    return layer_traj, unique_mols_allframes


# In[ ]:





# In[ ]:


def make_theta(traj, unique_mols):
    nsteps = len(traj)
    out = {mol_id: np.zeros(nsteps) for mol_id in unique_mols}
    for counter, time in enumerate(tqdm(traj)):
        frame = traj[time].copy(deep=True)
        frame_mols = frame['mols'].to_numpy()
        for mol_id in out:
            if mol_id in frame_mols:
                out[mol_id][counter] = 1
    return out


# In[ ]:


@njit
def get_nonzero_blocks(arr):
    """
    Written with ChatGPT. 
    
    Returns a list of blocks of continuous non-zero values in a 1D NumPy array.

    Args:
        arr (numpy.ndarray): A 1D NumPy array.

    Returns:
        list: A list of blocks of continuous non-zero values in the array.
              Each block is represented as a list of indices.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
        >>> get_nonzero_blocks(arr)
        [[0, 1, 2, 3], [9, 10, 11], [14, 15], [20, 21]]

    """
    blocks = []
    current_block = None
    for i in range(arr.size):
        x = arr[i]
        if x != 0:
            if current_block is None:
                current_block = [i]
            else:
                current_block.append(i)
        else:
            if current_block is not None:
                blocks.append(current_block)
                current_block = None
    if current_block is not None:
        blocks.append(current_block)
    return blocks


# In[ ]:


@njit
def get_nonzero_blocks_values(arr):
    """
    Written with ChatGPT. 
    
    Returns a list of blocks of continuous non-zero values in a 1D NumPy array.

    Args:
        arr (numpy.ndarray): A 1D NumPy array.

    Returns:
        list: A list of blocks of continuous non-zero values in the array.
              Each block is represented as a list of values.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
        >>> get_nonzero_blocks_values(arr)
        [[1, 1, 1, 1], [1, 1, 1], [1, 1], [1, 1]]

    """
    blocks = []
    current_block = None
    for i in range(arr.size):
        x = arr[i]
        if x != 0:
            if current_block is None:
                current_block = [x]
            else:
                current_block.append(x)
        else:
            if current_block is not None:
                blocks.append(current_block)
                current_block = None
    if current_block is not None:
        blocks.append(current_block)
    return blocks


# In[ ]:


def generate_mol_presence(theta):
    out = {}
    for molid in tqdm(theta):
        theta_mol = theta[molid]
        mol_nonzero_blocks = get_nonzero_blocks(theta_mol)    
        out[molid] = mol_nonzero_blocks
    return out


# In[ ]:


def get_N_0_tau_normalisation(theta):
    """
    Will get the time dependant normalisation
    """
    N_0_tau = np.zeros(10)
    for molid in tqdm(theta):
        theta_mol = theta[molid]
        mol_nonzero_blocks = get_nonzero_blocks_values(theta_mol)
        for set_ in mol_nonzero_blocks:         
            # Determine the maximum length of the arrays
            max_len = max(len(N_0_tau), len(set_))
            # Pad the arrays with zeros to make them the same length
            N_0_tau_set = np.pad(set_, (0, max_len - len(set_)), 'constant')
            N_0_tau = np.pad(N_0_tau, (0, max_len - len(N_0_tau)), 'constant')
            # Add the arrays using broadcasting
            N_0_tau += N_0_tau_set
    return N_0_tau


# In[ ]:


def make_traj(fulltraj, presence):
    out = {}
    key = 0
    for molid in tqdm(presence):
        frames_set = presence[molid]
        for set_ in frames_set:
            molset = []
            for counter in set_:
                time = 'time_' + str(counter)
                framedata = fulltraj[time].copy(deep=True)
                mol_data = framedata.loc[(framedata['mols'] == molid)]
                molset.append(mol_data)
            assert len(molset) != 0, "check"
            out[key] = molset
            key += 1  
    return out


# In[ ]:


layertrajectory = make_traj(lmp, presence)


# In[ ]:


len(layertrajectory)


# In[ ]:


#def make_traj(fulltraj, presence):
    out = {}
    counter, key = 0, 0
    for molid in tqdm(presence):
        frames_set = presence[molid]
        counter = 0
        for set_ in frames_set:
            molset = []
            while (counter <= (set_[-1])):
                time = 'time_' + str(counter)
                framedata = fulltraj[time].copy(deep=True)
                mol_data = framedata.loc[(framedata['mols'] == molid)]
                molset.append(mol_data)
                counter += 1
            assert len(molset) != 0, "check"
            out[key] = molset
            key += 1  
    return out


# In[ ]:


def vectors(outtraj, box):
    box = np.array(box)
    out = []
    for time in tqdm(outtraj):
        moldata = outtraj[time]
        OH1, OH2 = [], []
        for molframe in moldata:
            molframe_xyz = molframe[['x', 'y', 'z']].to_numpy()
            #print(molframe)
            OH1_r_ik = (molframe_xyz[1] - molframe_xyz[0]) - (box * ((molframe_xyz[1] - molframe_xyz[0])/box).round())
            OH2_r_ik = (molframe_xyz[2] - molframe_xyz[0]) - (box * ((molframe_xyz[2] - molframe_xyz[0])/box).round())
            #print(OH1_r_ik.shape)
            OH1_r_ik_mag = np.linalg.norm((OH1_r_ik))
            OH2_r_ik_mag = np.linalg.norm((OH2_r_ik))
            OH1_r_ik_norm = OH1_r_ik / OH1_r_ik_mag
            OH2_r_ik_norm = OH2_r_ik / OH2_r_ik_mag 
            OH1.append(OH1_r_ik_norm)
            OH2.append(OH2_r_ik_norm)
        out.append(OH1)
        out.append(OH2)
    return out


# In[ ]:


test = vectors(layertrajectory, box=[27.72, 27.72, 150.0])


# In[ ]:





# In[ ]:


def DoP2(veclist):
    P2_sum = np.empty(0)
    for index, vecs in enumerate(tqdm(veclist)):
        #print(index)
        one_vec = np.array(vecs)
        x, y, z = one_vec[:, 0], one_vec[:, 1], one_vec[:, 2]
        P2 = direct_correlate(x,y,z)
        # Determine the maximum length of the arrays
        max_len = max(len(P2), len(P2_sum))
        # Pad the arrays with zeros to make them the same length
        P2 = np.pad(P2, (0, max_len - len(P2)), 'constant')
        P2_sum = np.pad(P2_sum, (0, max_len - len(P2_sum)), 'constant')
        # Add the arrays using broadcasting
        P2_sum += P2
    return P2_sum


# In[ ]:





# In[ ]:


def Master():
    return None


# In[ ]:





# In[ ]:


43.06/2


# # Ewald

# In[ ]:


lmp = curate_traj(lmpxyz="RCR/nveEw.xyz", nmol=1024)


# In[ ]:


lmp_zero = center_boxcenter(lmp, boxcenter=[13.86, 13.86, 21.53])


# In[ ]:


dens, z_bin = density_profile(lmp_zero, atom_traj='O')


# In[ ]:


plt.plot(z_bin, dens)
plt.plot(z_binGT, densGT)
#save_arrays_to_file(z_bin, dens, file_name="RCR/EwDens.dat")
#save_arrays_to_file(z_binGT, densGT, file_name="RCR/GTDens.dat")


# In[ ]:


first_layer, uniquemols = select_layer_oxygens(lmp, z_hi=3, z_lo=0)


# In[ ]:


theta = make_theta(first_layer, uniquemols)


# In[ ]:


presence = generate_mol_presence(theta)


# In[ ]:


layertrajectory = make_traj(lmp, presence)


# In[ ]:


layervecs = vectors(layertrajectory, box=[27.72, 27.72, 43.06])


# In[ ]:


P2 = DoP2(layervecs)


# In[ ]:


P2_Ew_norm = get_N_0_tau_normalisation(theta)


# In[ ]:


80*0.2


# In[ ]:


plt.xlim((0,100))
plt.plot(P2[:]/(P2_Ew_norm*2)[:])
plt.plot(P2GT[:]/(P2_GT_norm*2)[:])


# In[ ]:


time = np.arange(len(P2))*0.2


# In[ ]:


save_arrays_to_file(time, P2[:]/(P2_Ew_norm*2)[:], file_name="RCR/Bulk-EwaldP2.dat")


# In[ ]:


timeGT = np.arange(len(P2GT))*0.2


# In[ ]:


save_arrays_to_file(timeGT, P2GT[:]/(P2_GT_norm*2)[:], file_name="RCR/Bulk-GTP2.dat")


# In[ ]:





# # GT

# In[ ]:


lmpGT = curate_traj(lmpxyz="RCR/nveGT.xyz", nmol=1024)


# In[ ]:


lmpGT_zero = center_boxcenter(lmpGT, boxcenter=[13.86, 13.86, 21.53])


# In[ ]:


densGT, z_binGT = density_profile(lmpGT_zero, atom_traj='O')


# In[ ]:


save_arrays_to_file(z_binGT, densGT, file_name="densGT.dat")


# In[ ]:


first_layerGT, uniquemolsGT = select_layer_oxygens(lmpGT_zero, z_hi=3, z_lo=0)


# In[ ]:


thetaGT = make_theta(first_layerGT, uniquemolsGT)


# In[ ]:


presenceGT = generate_mol_presence(thetaGT)


# In[ ]:


layertrajectoryGT = make_traj(lmpGT, presenceGT)


# In[ ]:


layervecsGT = vectors(layertrajectoryGT, box=[27.72, 27.72, 43.06])


# In[ ]:


P2GT = DoP2(layervecsGT)


# In[ ]:


P2_GT_norm = get_N_0_tau_normalisation(thetaGT)


# In[ ]:


plt.plot(P2GT[:50]/(P2_GT_norm*2)[:50])


# In[ ]:


time = np.arange(len(P2))


# In[ ]:


time_LR,P2LR = np.loadtxt("rotacf_new.dat",unpack=True)


# In[ ]:





# In[ ]:


plt.xlim((0, 50))
plt.plot(time[:]*0.2, P2/(P2_Ew_norm*2))
plt.plot(time_LR[:], P2LR[:])


# In[ ]:





# In[ ]:


box=[27.72, 27.72, 43.06]

