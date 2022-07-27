import matplotlib.pyplot as plt


def plot_stacked(symbols, df, k='_var', title=None):
    fig, axs = plt.subplots(len(symbols))
    if title is None:
        title = k

    fig.suptitle(title)

    for i,s in enumerate(symbols):
        key = s + k
        axs[i].plot(df[key])
        axs[i].yaxis.set_label_position("right")
        axs[i].set_ylabel(symbols[i])

    plt.show()