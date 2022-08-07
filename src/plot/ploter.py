import matplotlib.pyplot as plt


def plot_stacked(symbols, df, k='_var', title=None, skip=False):
    fig, axs = plt.subplots(len(symbols) - int(skip))
    if title is None:
        title = k

    fig.suptitle(title)

    for i,s in enumerate(symbols):
        key = s + k
        if k and s == k.split('_')[1]:
            continue
        axs[i].plot(df[key])
        axs[i].yaxis.set_label_position("right")
        axs[i].set_ylabel(symbols[i])

    plt.show()