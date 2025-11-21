from cmdstanpy import CmdStanModel
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"


def comparison_plot(approximator, data):
    fig, ax = plt.subplots(figsize=(16 / 2.5, 9 / 2.5))

    bf_samples = approximator.quick_and_dirty_sample({"y": data["y"]}, 1000)
    for k, v in bf_samples.items():
        bf_samples[k] = v.cpu()

    stan_samples = sample_stan(data)

    bf_quantiles = {}
    stan_quantiles = {}
    vars = ["hyper_mean", "hyper_std", "shared_std", "local_mean"]

    for var in ["hyper_mean", "hyper_std", "shared_std", "local_mean"]:
        bf_quantiles[var] = np.quantile(bf_samples[var], [0.05, 0.25, 0.75, 0.95], axis=1)
        stan_quantiles[var] = np.quantile(stan_samples[var], [0.05, 0.25, 0.75, 0.95], axis=1)

    x = np.arange(9, dtype=float)
    dx = 0.08

    for i, var in enumerate(vars[:-1]):
        ax.vlines(
            x[i] - dx,
            bf_quantiles[var][0, :, :],
            bf_quantiles[var][3, :, :],
            lw=4.5,
            color="#0B2447",
            alpha=0.4,
        )
        ax.vlines(
            x[i] - dx,
            bf_quantiles[var][1, :, :],
            bf_quantiles[var][2, :, :],
            lw=4.5,
            color="#0B2447",
            alpha=0.8,
            label="GraphicalApproximator" if i == 0 else None,
        )

        ax.vlines(
            x[i] + dx,
            stan_quantiles[var][0, :, :],
            stan_quantiles[var][3, :, :],
            lw=4.5,
            color="darkred",
            alpha=0.2,
        )
        ax.vlines(
            x[i] + dx,
            stan_quantiles[var][1, :, :],
            stan_quantiles[var][2, :, :],
            lw=4.5,
            color="darkred",
            alpha=0.5,
            label="Stan" if i == 0 else None,
        )

    for i in range(6):
        ax.vlines(
            x[i + 3] - dx,
            bf_quantiles["local_mean"][0, :, i, :],
            bf_quantiles["local_mean"][3, :, i, :],
            lw=4.5,
            color="#0B2447",
            alpha=0.4,
        )
        ax.vlines(
            x[i + 3] - dx,
            bf_quantiles["local_mean"][1, :, i, :],
            bf_quantiles["local_mean"][2, :, i, :],
            lw=4.5,
            color="#0B2447",
            alpha=0.8,
        )

        ax.vlines(
            x[i + 3] + dx,
            stan_quantiles["local_mean"][0, :, i, :],
            stan_quantiles["local_mean"][3, :, i, :],
            lw=4.5,
            color="darkred",
            alpha=0.2,
        )
        ax.vlines(
            x[i + 3] + dx,
            stan_quantiles["local_mean"][1, :, i, :],
            stan_quantiles["local_mean"][2, :, i, :],
            lw=4.5,
            color="darkred",
            alpha=0.5,
        )

    print(bf_quantiles["local_mean"].shape)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [r"$\mu_{\lambda}$", r"$\sigma_{\lambda}$", r"$\sigma$", *[rf"$\lambda_{i + 1}$" for i in range(6)]], size=16
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("value", size=14)

    fig.tight_layout()
    return fig


def sample_stan(data):
    model_path = Path(__file__).parents[0] / "two_level.stan"
    model = CmdStanModel(stan_file=str(model_path))

    stan_input = {"J": data["y"].shape[1], "N": data["y"].shape[2], "y": data["y"][0, :, :, 0]}
    fit = model.sample(data=stan_input, chains=4, parallel_chains=4, adapt_delta=0.95)

    samples = {}
    samples["hyper_mean"] = fit.stan_variable("hyper_mean").reshape(1, -1, 1)
    samples["hyper_std"] = fit.stan_variable("hyper_std").reshape(1, -1, 1)
    samples["shared_std"] = fit.stan_variable("shared_std").reshape(1, -1, 1)
    samples["local_mean"] = fit.stan_variable("local_mean")[np.newaxis, :, :, np.newaxis]

    # for var in ["hyper_mean", "hyper_std", "shared_std", "local_mean"]:
    #     samples[var] = fit.stan_variable(var)

    return samples
