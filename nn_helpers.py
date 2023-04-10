import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


def update_normal_prior(n: int, mu: float, sigma: float, n0: int, mu0: float, sigma0: float) -> tuple:
    """Updates a normal prior distribution with new observations.

    Args:
        n (int): The number of new observations.
        mu (float): The mean of the new observations.
        sigma (float): The standard deviation of the new observations.
        n0 (int): The number of prior observations.
        mu0 (float): The mean of the prior observations.
        sigma0 (float): The standard deviation of the prior observations.

    Returns:
        tuple: A tuple containing the updated mean and standard deviation of the
               posterior distribution.
    """
    inv_vars1 = n0 / np.power(sigma0, 2), n / np.power(sigma, 2)
    mu1 = np.average((mu0, mu), weights=inv_vars1)
    sigma1 = 1 / np.sqrt(np.sum(inv_vars1))
    return mu1, sigma1


def dnorm_B_minus_A(mu_a: float, sigma_a: float, mu_b: float, sigma_b: float):
    """Calculates the density of a normal distribution B minus A.

    Args:
        mu_a (float): The mean of distribution A.
        sigma_a (float): The standard deviation of distribution A.
        mu_b (float): The mean of distribution B.
        sigma_b (float): The standard deviation of distribution B.

    Returns:
        rv_frozen: A frozen random variable object representing the normal
                   distribution of B minus A.
    """
    dnorm = norm(loc=mu_b - mu_a,
                 scale=np.sqrt(sigma_a ** 2 + sigma_b ** 2))
    return dnorm


def explain_prob_B_higher(df: pd.DataFrame, i: int) -> None:
    """Visualizes and explains the probability of variant B performing higher than variant A.

    Args:
        df (pd.DataFrame): The data frame containing the experiment results.
        i (int): The index of the experiment to analyze.

    Returns:
        None: The function does not return a value, but rather visualizes and prints
              the results of the experiment analysis.
    """
    mu = df['dnorm'][i].mean()
    sigma = df['dnorm'][i].std()

    x_axis = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    y_axis = df['dnorm'][i].pdf(x_axis)

    plt.title(df['metric'][i])
    plt.xlabel('norm_B - norm_A')
    plt.ylabel('Probability density')
    plt.plot(x_axis, y_axis, color='blue')
    plt.fill_between(x_axis, y_axis, color='blue', alpha=.1)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.show()

    avg_A = df['avg_A'][i]
    std_A = df['std_A'][i]
    avg_B = df['avg_B'][i]
    std_B = df['std_B'][i]
    D = avg_B - avg_A
    prob_B_higher = df['prob_B_higher'][i]

    print('Variant A performed avg_A = ' +
          '{:.4f}'.format(avg_A) + ', std_A = ' + '{:.4f}'.format(std_A) + ' vs avg_B = ' +
          '{:.4f}'.format(avg_B) + ', std_B = ' + '{:.4f}'.format(std_B) + '.')
    if df['avg_B'][i] > df['avg_A'][i]:
        print('avg_B is +' + '{:.4f}'.format(D) + ' higher than avg_A.')
        print('You can be ' + '{:.0%}'.format(prob_B_higher) +
              ' confident that this is a result of the changes you made and not a result of random chance.')
    elif df['avg_B'][i] < df['avg_A'][i]:
        print('avg_B is -' + '{:.4f}'.format(-D) + ' lower than avg_A.')
        print('You can be ' + '{:.0%}'.format(1 - prob_B_higher) +
              ' confident that this is a result of the changes you made and not a result of random chance.')
    else:
        print('There is no statistical difference between variants A and B.')
