import matplotlib.pyplot as plt
import numpy as np


def plot_graphs(
    histories,
    labels,
    n,
    m,
    mu_str,
    colors,
    linewidths,
    linestyles,
    n_iters=None,
    threshold=1e-8,
    filename=None,
    f_star=None,
    suptitle=None,
    max_iter=1000,
):

    if f_star is None:
        f_best = min(histories[0]["func"])
        for i in range(1, len(histories)):
            f_best = min(f_best, min(histories[i]["func"]))
    else:
        f_best = f_star

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    for i in range(len(histories)):
        resid = np.array(histories[i]["func"]) - f_best
        n_iter = n_iters[i] if n_iters is not None else resid.shape[0]
        n_iter = min(n_iter, np.searchsorted(-resid, -threshold) + 1)

        ax1.semilogy(
            resid[0 : min(n_iter, max_iter)],
            label=labels[i],
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=linewidths[i],
        )
        ax2.semilogy(
            histories[i]["time"][0:n_iter],
            resid[0:n_iter],
            label=labels[i],
            linestyle=linestyles[i],
            color=colors[i],
            linewidth=linewidths[i],
        )

    ax1.set_xlabel("Iterations", fontsize=14)
    ax1.set_ylabel("Func. residual", fontsize=14)
    ax2.set_xlabel("Time", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid()
    ax2.grid()
    # plt.tight_layout()
    if filename:
        print("output: %s" % filename)
        plt.savefig(filename)


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
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
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
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
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
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
    plt.show()