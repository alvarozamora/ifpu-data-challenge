# from astropy.io import fits
import numpy as np
import scipy.spatial
# from sklearn.neighbors import KDTree, BallTree
from scipy.stats import poisson, erlang
from scipy import interpolate
from os import urandom
import struct
# from astropy.cosmology import FlatLambdaCDM
# from astropy import units as u
# from astropy.coordinates import SkyCoord
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['grid.color'] = 'grey'
matplotlib.rcParams['grid.linestyle'] = '--'
matplotlib.rcParams['grid.linewidth'] = 0.4
matplotlib.rcParams['grid.alpha'] = 0.5

# fig = plt.figure()
# from matplotlib.ticker import AutoMinorLocator, LogLocator
# cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
# h = 0.6774
# baseDir = '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/redMapper'

################################

def VolumekNN(xin, xout, k=1, periodic = 0):
    if isinstance(k, int): k = [k] # 
    dim = xin.shape[1] # dimension of every row
    #Ntot = xin.shape[0] # dimension of entries (length of column)
    xtree = scipy.spatial.cKDTree(xin, boxsize=periodic)
    #print('k = ', k)
    dis, disi = xtree.query(xout, k=k, n_jobs=8) # dis is the distance to the kth nearest neighbour, disi is the id of that neighbour
    vol = np.empty_like(dis) # same shape as distance including all k values
    Cr = [2, np.pi, 4 * np.pi / 3, np.pi**2, 8*np.pi**2/15][dim - 1]  # Volume prefactor for 1,2, 3D
    for c, k in enumerate(np.nditer(np.array(k))):
        #print('c, dim, dis = ', c, dim, dis[:, c]**dim / k)
        vol[:, c] = Cr * dis[:, c]**dim / k # the overdense average volume per point in sphere
        #print('vol = ', vol[:, c])
    return vol

def CDFVolkNN(vol): # CDF
    CDF = []
    N = vol.shape[0]
    l = vol.shape[1]
    gof = ((np.arange(0, N) + 1) / N)
    for c in range(l):
        ind = np.argsort(vol[:, c])
        sVol= vol[ind, c]
        # return array of interpolating functions
        CDF.append(interpolate.interp1d(sVol, gof, kind = 'linear', \
                                        bounds_error=False)) # x = sVol, y = gof
    return CDF


################################

# bine = np.logspace(np.log10(30), np.log10(220), 51)
# binw = bine[1:] - bine[:-1]
# binc = (bine[1:] + bine[:-1]) / 2

# vol_h = np.loadtxt(baseDir+'/Dis_h_3D_Y500.txt') # Reads in r kNN data
# CDFs_h = CDFVolkNN(vol_h)
# CDF_1h = interpolate.interp1d(binc, CDFs_h[0](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
# CDF_2h = interpolate.interp1d(binc, CDFs_h[4](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
# CDF_3h = interpolate.interp1d(binc, CDFs_h[5](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
# CDF_4h = interpolate.interp1d(binc, CDFs_h[6](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))

# Yd1 = CDF_1h(binc)
# Yd2 = CDF_2h(binc)
# Yd3 = CDF_3h(binc)
# Yd4 = CDF_4h(binc)

# print(Yd1)
# print(Yd2)
# print(Yd3)
# print(Yd4)

# np.savetxt(baseDir+'/C43/CDF_1NN.txt', Yd1)
# np.savetxt(baseDir+'/C43/CDF_2NN.txt', Yd2)
# np.savetxt(baseDir+'/C43/CDF_3NN.txt', Yd3)
# np.savetxt(baseDir+'/C43/CDF_4NN.txt', Yd4)

def CDF_2nn(CDF_1NN):
    c2 = CDF_1NN + (1-CDF_1NN) * np.log(1-CDF_1NN)
    return c2

def CDF_3NN(CDF_1NN, CDF_2NN):
    c3 = CDF_2NN + ( (1-CDF_1NN)*np.log(1-CDF_1NN) + (CDF_1NN - CDF_2NN) - 
                    1/2*(CDF_1NN - CDF_2NN)**2/(1-CDF_1NN) )
    return c3

def CDF_4NN(CDF_1NN, CDF_2NN, CDF_3NN):
    c4 = CDF_3NN + (CDF_1NN - CDF_2NN)/(1 - CDF_1NN) * ( (1-CDF_1NN)*np.log(1-CDF_1NN) + (CDF_1NN - CDF_2NN)
                                                        - 1/6 * (CDF_1NN-CDF_2NN)**2/(1-CDF_1NN))
    return c4

# C2 = CDF_2nn(Yd1)
# C3 = CDF_3NN(Yd1, Yd2)
# C41 = CDF_4NN(Yd1, Yd2, C3)
# C42 = CDF_4NN(Yd1, Yd2, Yd3)
# print(C3)
# print(C41)
# print(C42)

def get_pCDF(cdf):
    id_rise = np.where(cdf <= 0.5)[0]
    id_drop = np.where(cdf > 0.5)[0]
    pcdf = np.concatenate((cdf[id_rise], 1-cdf[id_drop]))
    return pcdf

# pCDF_1 = get_pCDF(Yd1)
# pCDF_2 = get_pCDF(Yd2)
# pCDF_3 = get_pCDF(Yd3)
# pCDF_4 = get_pCDF(Yd4)
# pCDF_C2 = get_pCDF(C2)
# pCDF_C3 = get_pCDF(C3)
# pCDF_C41 = get_pCDF(C41)
# pCDF_C42 = get_pCDF(C42)

# ################################
# jack = 200
# for i in range(jack):
#     print(i)
#     vol_h = np.loadtxt(baseDir+'/Dis_jack_{}.txt'.format(i))
    
#     CDFs_h = CDFVolkNN(vol_h)
#     CDF_1h = interpolate.interp1d(binc, CDFs_h[0](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
#     CDF_2h = interpolate.interp1d(binc, CDFs_h[4](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
#     CDF_3h = interpolate.interp1d(binc, CDFs_h[5](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
#     CDF_4h = interpolate.interp1d(binc, CDFs_h[6](binc), kind='linear', bounds_error=False, fill_value=(0.,1.))
    
#     cdf1 = CDF_1h(binc)
#     cdf2 = CDF_2h(binc)
#     cdf3 = CDF_3h(binc)
#     cdf4 = CDF_4h(binc)
    
#     np.savetxt(baseDir+'/C43/jack_{}_1NN.txt'.format(i), cdf1)
#     np.savetxt(baseDir+'/C43/jack_{}_2NN.txt'.format(i), cdf2)
#     np.savetxt(baseDir+'/C43/jack_{}_3NN.txt'.format(i), cdf3)
#     np.savetxt(baseDir+'/C43/jack_{}_4NN.txt'.format(i), cdf4)









###
if __name__ == "__main__":

    subsamples = [250, 500, 1000]
    subsamples = [250, 500]
    YMIN = 1e-5
    PLOT_YMIN = 1e-3
    PERCENTILES = np.sort(np.append(np.logspace(np.log10(YMIN), np.log10(0.5), 400), 1-np.logspace(np.log10(YMIN), np.log10(0.5), 400)[:-1]))
    PCDF = np.minimum(PERCENTILES, 1-PERCENTILES)

    cmap=plt.get_cmap('tab20')

    for sub in subsamples:

        fig1, ax1 = plt.subplots(figsize=(12,8))
        fig2, [ax2_1, ax2_2] = plt.subplots(2, 1, figsize=(12,16))

        data_sub = np.load(f"challenge0_{sub}.npz")
    
        for run in range(1,15+1):

            key1 = f"run{run}_1"
            key2 = f"run{run}_2"
            key3 = f"run{run}_3"
            key4 = f"run{run}_4"

            cdf1 = data_sub[key1]
            cdf2 = data_sub[key2]
            cdf3 = data_sub[key3]
            cdf4 = data_sub[key4]

            min_dist = cdf3.min()
            max_dist = cdf1.max()
            dist = np.logspace(np.log10(min_dist), np.log10(max_dist), 200)
            interp_cdf1 = np.interp(dist, cdf1, PERCENTILES)
            interp_cdf2 = np.interp(dist, cdf2, PERCENTILES)
            pred_cdf3 = CDF_3NN(interp_cdf1, interp_cdf2)
            pred_cdf4 = CDF_4NN(interp_cdf1, interp_cdf2, pred_cdf3)
            pred_pcdf3 = np.minimum(pred_cdf3, 1-pred_cdf3)
            pred_pcdf4 = np.minimum(pred_cdf4, 1-pred_cdf4)

            ax1.loglog(cdf1, PCDF, label="1NN")
            ax1.loglog(cdf2, PCDF, label="2NN")
            ax1.loglog(cdf3, PCDF, label="3NN")
            ax1.loglog(cdf4, PCDF, label="4NN")
            ax1.loglog(dist, pred_pcdf3, "--", label="pred 3NN")
            ax1.loglog(dist, pred_pcdf4, "--", label="pred 4NN")

            ax1.set_ylim(PLOT_YMIN)
            ax1.set_xlabel("Distance (Mpc/h)")
            ax1.set_ylabel("Peaked CDF")
            ax1.set_title(rf"Peaked kNNs for $n = {sub}$")
            ax1.grid(alpha=0.6)
            ax1.legend()
            fig1.savefig(f"knns_{sub}_run{run}", dpi=230)
            ax1.cla()


            ax2_1.loglog(cdf3, np.abs(1-np.interp(cdf3, dist, pred_cdf3, right=np.nan)/PERCENTILES), label=str(run), color=cmap(run))
            ax2_1.set_xlabel("Distance (Mpc/h)")
            ax2_1.set_ylabel(r"$|1-\mathrm{3NN predicted/measured}|$")
            ax2_1.grid(alpha=0.6)

            ax2_2.loglog(cdf4, np.abs(1-np.interp(cdf4, dist, pred_cdf4, right=np.nan)/PERCENTILES), label=str(run), color=cmap(run))
            ax2_2.set_xlabel("Distance (Mpc/h)")
            ax2_2.set_ylabel(r"$|1-\mathrm{4NN predicted/measured}|$")
            ax2_2.grid(alpha=0.6)
            

        ax2_1.legend()
        ax2_2.legend()
        fig2.suptitle(rf"Predicted/Measured for $n = {sub}$")
        fig2.savefig(f"pred_measured_{sub}.png", dpi=230)
        fig2.clf()

            


