# code adapted from answer 2 by e-malito
# https://stackoverflow.com/questions/43150872/number-of-arrowheads-on-matplotlib-streamplot

import numpy as np


if __name__ == "__main__":
    # --- Main body of streamQuiver
    # Extracting lines
    import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # sp = ax.streamplot(Grid[:, 0].reshape((50, 50)),
    #                    Grid[:, 1].reshape((50, 50)),
    #                    VF[:, 0].reshape((50, 50)),
    #                    VF[:, 1].reshape((50, 50)), arrowstyle='-', density=10)
    #
    # streamQuiver(ax, sp, n=3)
