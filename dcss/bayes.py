import pandas as pd
import numpy as np
import seaborn as sns
import pymc as pm
import arviz as az

import matplotlib as mpl
from matplotlib import pyplot as plt

from .style import set_style

set_style()


def plot_2020_election_diff(df):
    fig, ax = plt.subplots(figsize=(8, 7))
    g = sns.scatterplot(x=df['spend'], y=df['vote'], alpha=.8)

    g.set(xlim=(-2.5e7, 15000000))
    g.set(ylim=(-250000, 350000))

    plt.text(-2e7, 300000, "Democrats Win and Underspend")
    plt.text(-2e7, -200000, "Democrats Lose and Underspend")
    plt.text(3e6, -200000, "Democrats Lose and Overspend")
    plt.text(3e6, 300000, "Democrats Win and Overspend")

    plt.axhline(y=0, color='grey')
    plt.axvline(x=0, color='grey')

    plt.axhspan(0, 370000, xmin=0.625, xmax=1, facecolor='gray', alpha=0.3)
    plt.axhspan(0, -370000, xmin=0, xmax=0.625, facecolor='crimson', alpha=0.1)
    plt.axhspan(0, 370000, xmin=0, xmax=0.625, facecolor='lightgray', alpha=0.3)
    plt.axhspan(0, -370000, xmin=0.625, xmax=1, facecolor='crimson', alpha=0.3)

    # Style the axes
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set(xlabel='Spending differential (Democrat - Republican)',
           ylabel='Vote differential (Democrat - Republican)')

    sns.despine(left=True, bottom=True)
    plt.show()


def plot_2020_election_fit(spend_std, vote_std, trace_pool, ppc):
    g = sns.scatterplot(x=spend_std, y=vote_std, alpha=.8)
    g.set(xlim=(-10, 5))
    g.set(ylim=(-3, 4))
    g.axhline(y=0, color='grey')
    g.axvline(x=0, color='grey')
    x_range = np.linspace(-10, 4, 10)

    # Access the variables from the 'posterior' group
    alpha_samples = trace_pool.posterior['alpha'].values.flatten()
    beta_samples = trace_pool.posterior['beta'].values.flatten()
    alpha_m = alpha_samples.mean()
    beta_m = beta_samples.mean()

    g.plot(
        x_range,
        alpha_m + beta_m * x_range,  # This is our linear model
        c='k',
    )

    # Convert spend_std to a NumPy array
    spend_std_np = np.array(spend_std)

    # Compute mu_pool
    mu_pool = (
        alpha_samples[:, np.newaxis] + beta_samples[:, np.newaxis] * spend_std_np[np.newaxis, :]
    )

    az.plot_hdi(
        spend_std,
        mu_pool,
        ax=g,
        fill_kwargs={"alpha": 0.4, "label": "Mean outcome 94% HPD"},
    )

    # Get votes samples
    votes_samples = ppc.posterior_predictive['votes'].values
    votes_samples = votes_samples.reshape(-1, votes_samples.shape[-1])  # (n_samples, n_observations)

    az.plot_hdi(
        spend_std,
        votes_samples,
        ax=g,
        fill_kwargs={"alpha": 0.4, "color": "lightgray", "label": "Outcome 94% HPD"}
    )

    sns.despine()
    plt.show()



def plot_2020_partial_pool(
    partial_pool_model_regularized,
    trace_partial_pool_regularized,
    trace_no_pool,
    n_states,
    state_idx,
    spend_std,
    vote_std,
    ppc,
    state_cat
):
    _, ax = plt.subplots(
        8,
        6,
        figsize=(12, 16),
        constrained_layout=True
    )

    # Flatten the array from 'ax' to make iterating easier
    ax = np.ravel(ax)

    # Define a range of values to plot our estimator line on
    x_range = np.linspace(-8, 4, 10)

    # Access alpha and beta samples from trace_partial_pool_regularized
    alpha_samples = trace_partial_pool_regularized.posterior['alpha'].values  # shape (chain, draw, n_states)
    beta_samples = trace_partial_pool_regularized.posterior['beta'].values     # shape (chain, draw, n_states)

    # Access votes samples from ppc
    votes_samples = ppc.posterior_predictive['votes'].values  # shape (chain, draw, observation)
    votes_samples = votes_samples.reshape(-1, votes_samples.shape[-1])  # (n_samples, n_observations)

    for i in range(n_states):
        ax[i].set_xlim((-4, 4))
        ax[i].set_ylim((-4, 4))

        # Create a scatterplot of the data from each state
        ax[i].scatter(spend_std[state_idx == i], vote_std[state_idx == i])

        # Compute mean alpha and beta for state i
        alpha_m = alpha_samples[:, :, i].mean()
        beta_m = beta_samples[:, :, i].mean()

        ax[i].plot(
            x_range,
            alpha_m + beta_m * x_range,  # This is our linear model
            c='k',
        )

        if len(spend_std[state_idx == i]) > 1:
            # Get samples for state i
            alpha_i_samples = alpha_samples[:, :, i].reshape(-1)
            beta_i_samples = beta_samples[:, :, i].reshape(-1)

            # Compute mu_pp for state i
            spend_state = np.array(spend_std[state_idx == i])  # Convert to NumPy array
            mu_pp = (
                alpha_i_samples[:, np.newaxis]
                + beta_i_samples[:, np.newaxis] * spend_state[np.newaxis, :]
            )

            az.plot_hdi(
                spend_state,
                mu_pp,
                ax=ax[i],
                fill_kwargs={"alpha": 0.4, "label": "Mean outcome 94% HPD"},
            )

            # Get votes_samples for observations in state i
            votes_state_samples = votes_samples[:, state_idx == i]

            az.plot_hdi(
                spend_state,
                votes_state_samples,
                ax=ax[i],
                fill_kwargs={"alpha": 0.4, "color": "lightgray", "label": "Outcome 94% HPD"}
            )

        ax[i].set_title(state_cat.categories[i])


def plot_2020_no_pool(
    no_pool_model,
    trace_no_pool,
    n_states,
    state_idx,
    spend_std,
    vote_std,
    ppc,
    state_cat
):
    # Initialize one subplot for each state
    _, ax = plt.subplots(
        8,
        6,
        figsize=(12, 16),
        constrained_layout=True
    )

    # Flatten the array from 'ax' to make iterating easier
    ax = np.ravel(ax)

    # Define a range of values to plot our estimator line on
    x_range = np.linspace(-8, 4, 10)

    with no_pool_model:
        for i in range(n_states):
            ax[i].set_xlim((-4, 4))
            ax[i].set_ylim((-4, 4))

            # Create a scatterplot of the data from each state
            state_mask = state_idx == i
            ax[i].scatter(spend_std[state_mask], vote_std[state_mask])

            alpha_m = trace_no_pool.posterior['alpha'].values[:, :, i].mean()
            beta_m = trace_no_pool.posterior['beta'].values[:, :, i].mean()

            ax[i].plot(
                x_range,
                alpha_m + beta_m * x_range,  # This is our linear model
                c='k',
            )

            if len(spend_std[state_mask]) > 1:
                mu_pp = (
                    trace_no_pool.posterior['alpha'].values[:, :, i].mean(axis=(0, 1))
                    + trace_no_pool.posterior['beta'].values[:, :, i].mean(axis=(0, 1)) * np.array(spend_std[state_mask])[:, None]
                )

                az.plot_hdi(
                    spend_std[state_mask],
                    mu_pp.T,
                    ax=ax[i],
                    fill_kwargs={"alpha": 0.4, "label": "Mean outcome 94% HPD"},
                )

                # Ensure the dimensions match for votes samples
                votes_state_samples = ppc.posterior_predictive['votes'].values[:, :, state_mask].reshape(-1, len(spend_std[state_mask]))

                az.plot_hdi(
                    spend_std[state_mask],
                    votes_state_samples,
                    ax=ax[i],
                    fill_kwargs={"alpha": 0.4, "color": "lightgray", "label": "Outcome 94% HPD"}
                )

            ax[i].set_title(state_cat.categories[i])