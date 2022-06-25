import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = "white"
from scipy.spatial import cKDTree as Tree
import glob
from tqdm import tqdm

# Simulation Parameter
BOXSIZE = 2000

# kNN Parameters
NRAND = 10**5
YMIN = 1e-3
PERCENTILES = np.sort(np.append(np.logspace(np.log10(YMIN), np.log10(0.5), 100), 1-np.logspace(np.log10(YMIN), np.log10(0.5), 100)[:-1]))
print(PERCENTILES)
PCDF = np.minimum(PERCENTILES, 1-PERCENTILES)
k = [1,2,3,4]; kstr = [str(kk) for kk in k]

# Number Density for Subsample (count per (Gpc/h)^3)
SUBSAMPLE = 10**3

# Grab all runs for this challenge
runs = list(np.sort(glob.glob("run*.npz")))

results = []
plt.figure(figsize=(12*len(k),8))
for run in runs:
    
    # Load catalog for this run
    catalog = np.load(run)

    # Unpack position
    pos = catalog['pos']

    # Shuffle and chunk positions (The 8 is here because this is a (2 Gpc/h)^3 box)
    NUM_SUBSAMPLES = len(pos) // SUBSAMPLE // 8
    pos_array = np.split(np.random.permutation(pos)[:8 * SUBSAMPLE * NUM_SUBSAMPLES], NUM_SUBSAMPLES)

    # Accumulator for the subsample results
    run_results = []

    # Iterator
    iterator = tqdm(pos_array)
    iterator.set_description(run.replace(".npz",""))
    for pos in iterator:

        # Generate tree for this run
        tree = Tree(pos, leafsize=32, compact_nodes=True, balanced_tree=True, boxsize=BOXSIZE)

        # Generate random queries and query 1NNs
        queries = np.random.uniform(size=(NRAND, 3)) * BOXSIZE
        r, ids = tree.query(queries, k=k, workers=16)

        # Get subsample result (percentiles)
        sub_result = np.percentile(r, PERCENTILES * 100, axis=0)

        # Append subsample result to run results
        run_results.append(sub_result)
    
    # Append mean of all subsamples to results
    results.append(np.mean(run_results, axis=0))

# Mean across all runs
mean_result = np.mean(results, axis=0)


def plot_pCDF():

    for (j, kk) in enumerate(k):
        plt.subplot(1, len(k), j+1)
        for (i, run) in enumerate(runs):

            result = results[runs.index(run)]
            plt.loglog(result[:,j], PCDF, '.', label=run.replace(".npz", ""))

            if i == len(runs)-1:
                plt.title(f"{kk}NN Peaked CDF")
                plt.ylabel(f"{kk}NN Peaked CDF")
                plt.xlabel("Distance")
                plt.legend(ncol=2)
                plt.grid(alpha=0.4)

    plt.suptitle(rf"Peaked CDFs for $n = {SUBSAMPLE}$ ($h$/Gpc)$^3$")
    plt.savefig("challenge1.png")
    plt.clf()


def plot_mean_residual():

    for (j, kk) in enumerate(k):
        plt.subplot(1, len(k), j+1)
        for (i, run) in enumerate(runs):
    
            result = results[runs.index(run)]
            plt.plot(PERCENTILES, mean_result[:,j] - result[:,j], label=run.replace(".npz", ""))

            if i == len(runs)-1:
                plt.title("Residual with respect to mean at every percentile")
                plt.ylabel("Residual")
                plt.xlabel("Percentile")
                plt.legend(ncol=2)
                plt.grid(alpha=0.4)

    plt.suptitle(rf"Residual with respect to mean at every percentile for $n = {SUBSAMPLE}$ ($h$/Gpc)$^3$")
    plt.savefig("challenge1_mean_residual.png")
    plt.clf()


def save_results():

    # Initialize dictionary
    results_to_save = {}

    # Save parameters
    results_to_save["num_density"] = np.array([SUBSAMPLE])
    results_to_save["nrand_per_subsample"] = np.array([NRAND])

    # Populate dictionary with (key, value) pairs
    for run in runs:

        # Find index of run
        idx = runs.index(run)

        for (j, kk) in enumerate(k):

            # Get (key, value) pair
            key = f"{run.replace('.npz', '')}_{kk}"
            value = results[idx][:,j]

            # Store (key, value) pair in the dictionary
            results_to_save[key] = value

    # Save dictionary to disk with SUBSAMPLE as key
    np.savez("challenge1.npz", **results_to_save)


plot_pCDF()
plot_mean_residual()
save_results()









    


