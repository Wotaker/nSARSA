from __future__ import annotations

from matplotlib import pyplot as plt
import numpy as np
import os
from lz4.frame import compress, decompress
import pickle

import problem


def save(obj, filename):
    serialized_obj = pickle.dumps(obj)
    compressed_obj = compress(serialized_obj)
    with open(filename, 'wb') as f:
        f.write(compressed_obj)


def load(filename):
    with open(filename, 'rb') as f:
        compressed_obj = f.read()
    decompressed_obj = decompress(compressed_obj)
    obj = pickle.loads(decompressed_obj)
    return obj


def draw_arrow(axes: plt.Axes, begin: tuple[int, int], end: tuple[int, int]) -> None:
    (begin_y, begin_x), (end_y, end_x) = begin, end
    delta_x, delta_y = end_x - begin_x, end_y - begin_y
    axes.arrow(begin_x + 0.5, begin_y + 0.5, delta_x, delta_y,
               length_includes_head=True,
               head_width=0.8, head_length=0.8,
               fc='r', ec='r')


def draw_episode(track: np.ndarray, positions: list[problem.Position], episode: int, save_prefix: str) -> None:
    
    # if save_prefix directory does not exist, create it
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)

    # Plot the track
    ax = plt.axes()
    ax.imshow(track)
    for i in range(len(positions) - 1):
        begin, end = positions[i], positions[i+1]
        draw_arrow(ax, begin, end)
    plt.savefig(f'{save_prefix}/track_{episode}.png', dpi=300)
    plt.clf()


def draw_penalties_plot(penalties: list[int], window_size: int, episode: int, save_prefix: str) -> None:
    
    # if save_prefix directory does not exist, create it
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)

    # Plot the penalties
    means = [np.mean(penalties[i:i+window_size]) for i in range(len(penalties) - window_size)]
    ax = plt.axes()
    ax.plot(means)
    plt.savefig(f'{save_prefix}/penalties.png', dpi=300)
    plt.clf()
