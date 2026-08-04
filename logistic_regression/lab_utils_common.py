import numpy as np
import matplotlib.pyplot as plt


def draw_vthresh(ax, x, threshold):
    """
    Draw a vertical threshold line on the given axes.

    Parameters:
        ax        : matplotlib axes
        x         : x-coordinate of threshold
        threshold : threshold value
    """
    ax.axvline(
        x=threshold,
        linestyle='--',
        linewidth=1
    )