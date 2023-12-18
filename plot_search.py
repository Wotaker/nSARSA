import os
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_DIR = "/Users/wciezobka/agh/SemestrIX/uczenie-maszynowe/laby/lab03/nSARSA/out/nepisodes5000-mapc"

def extract_alphas():
    alphas = []
    for file_name in os.listdir(DEFAULT_DIR):
        if file_name.endswith(".npy"):
            alpha = float(file_name.split("-")[2][1:][:-4])
            alphas.append(alpha)
    return sorted(list(set(alphas)))

def extract_ns():
    ns = []
    for file_name in os.listdir(DEFAULT_DIR):
        if file_name.endswith(".npy"):
            n = int(file_name.split("-")[1][1:])
            ns.append(n)
    return sorted(list(set(ns)))

def plot_search():

    fig, ax = plt.subplots(figsize=(6, 6))
    alphas = extract_alphas()
    for n_step in extract_ns():
        estimated_regrets = []
        alpha_values = []
        for alpha in alphas:
            file_name = f"regret-n{n_step}-a{alpha}.npy"
            try:
                penalty = np.load(os.path.join(DEFAULT_DIR, file_name))
                estimated_regret = -np.sum(penalty)
                estimated_regrets.append(estimated_regret)
                alpha_values.append(alpha)
            except FileNotFoundError:
                continue
        ax.scatter(alpha_values, estimated_regrets, label=f"n={n_step}", s=10)
    ax.set_xlabel("alpha")
    ax.set_ylabel("Estimated Regret")
    ax.set_yscale('log')
    ax.legend()
    plt.savefig("map_c-search-results.png", bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    plot_search()
