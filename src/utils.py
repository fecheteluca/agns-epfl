import matplotlib.pyplot as plt
import numpy as np


def plot_only_iterations(
    histories,
    labels,
    f_star,
    colors,
    linewidths,
    linestyles,
    threshold=1e-8,
    max_iter=1000,
    save_as=None,
    title=None,
):
    plt.figure(figsize=(10, 6))
    for i, hist in enumerate(histories):
        resid = np.array(hist["func"]) - f_star
        last = np.searchsorted(-resid, -threshold) + 1
        iters = min(len(resid), last, max_iter)

        plt.semilogy(
            np.arange(iters),
            resid[:iters],
            linestyle=linestyles[i],
            linewidth=linewidths[i],
            color=colors[i],
            label=labels[i],
        )

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Iterations", fontsize=20)
    plt.ylabel(r"$f(x_k) - f^\star$", fontsize=20)
    if title:
        plt.title(title, fontsize=22, y=1.02)
    plt.legend(fontsize=18)
    if save_as:
        plt.savefig(save_as)
    plt.show()


def plot_only_time(
    histories,
    labels,
    f_star,
    colors,
    linewidths,
    linestyles,
    threshold=1e-8,
    max_iter=1000,
    save_as=None,
    title=None,
):
    plt.figure(figsize=(10, 6))
    for i, hist in enumerate(histories):
        func = np.array(hist["func"])
        time = np.array(hist["time"])
        resid = func - f_star
        last = np.searchsorted(-resid, -threshold) + 1
        n_pts = min(len(resid), last, max_iter)

        plt.semilogy(
            time[:n_pts],
            resid[:n_pts],
            linestyle=linestyles[i],
            linewidth=linewidths[i],
            color=colors[i],
            label=labels[i],
        )

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel(r"$f(x_k) - f^\star$", fontsize=20)
    if title:
        plt.title(title, fontsize=22, y=1.02)
    plt.legend(fontsize=18)
    if save_as:
        plt.savefig(save_as)
    plt.show()


def plot_only_operations(
    histories,
    labels,
    f_star,
    operation_key,
    colors,
    linewidths,
    linestyles,
    threshold=1e-8,
    max_ops=None,
    save_as=None,
    title=None,
):
    plt.figure(figsize=(10, 6))
    for i, hist in enumerate(histories):
        func = np.array(hist["func"])
        ops = np.array(hist[operation_key])
        resid = func - f_star

        last = np.searchsorted(-resid, -threshold) + 1
        n_pts = min(len(resid), last)
        if max_ops is not None:
            valid = ops[:n_pts] <= max_ops
            n_pts = valid.sum()

        plt.semilogy(
            ops[:n_pts],
            resid[:n_pts],
            linestyle=linestyles[i],
            linewidth=linewidths[i],
            color=colors[i],
            label=labels[i],
        )

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(operation_key.replace("_", " ").title(), fontsize=20)
    plt.ylabel(r"$f(x_k) - f^\star$", fontsize=20)
    if title:
        plt.title(title, fontsize=22, y=1.02)
    plt.legend(fontsize=18)
    if save_as:
        plt.savefig(save_as)
    plt.show()