import numpy as np
import pandas as pd
from scipy.stats import beta, norm
import matplotlib.pyplot as plt


def update_beta_prior(trials: int, success: int, alpha0: float, beta0: float) -> tuple[float, float]:
    """
    Updates the prior beta distribution parameters, alpha and beta, based on the number of trials and successes.

    Args:
        trials (int): The number of trials.
        success (int): The number of successes.
        alpha0 (float): The prior alpha parameter.
        beta0 (float): The prior beta parameter.

    Returns:
        A tuple containing the updated alpha and beta parameters, alpha1 and beta1 respectively.
    """
    alpha1 = alpha0 + success
    beta1 = beta0 + trials - success
    return alpha1, beta1


def dbeta_B_minus_A(alpha_a: float, beta_a: float, alpha_b: float, beta_b: float) -> norm:
    """
    Calculates the difference between two beta distributions, B and A, and returns a normal distribution.

    Args:
        alpha_a (float): The alpha parameter of distribution A.
        beta_a (float): The beta parameter of distribution A.
        alpha_b (float): The alpha parameter of distribution B.
        beta_b (float): The beta parameter of distribution B.

    Returns:
        A normal distribution representing the difference between distributions B and A.
    """
    beta_mean_a = beta.mean(alpha_a, beta_a)
    beta_mean_b = beta.mean(alpha_b, beta_b)
    beta_var_a = beta.var(alpha_a, beta_a)
    beta_var_b = beta.var(alpha_b, beta_b)

    dbeta = norm(loc=beta_mean_b - beta_mean_a,
                 scale=np.sqrt(beta_var_a + beta_var_b))
    return dbeta


def explain_prob_B_higher(df: pd.DataFrame, i: int) -> None:
    """
    Plots the probability density function of the difference between two beta distributions and prints a summary of the
    results for a specific index in a pandas DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the beta distribution parameters, p-values, and probability of
                           B being higher for each variant.
        i (int): The index of the variant to summarize and plot.

    Returns:
        None. The function only produces a plot and prints a summary.
    """
    mu = df['dbeta'][i].mean()
    sigma = df['dbeta'][i].std()

    x_axis = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    y_axis = df['dbeta'][i].pdf(x_axis)

    plt.title(df['metric'][i])
    plt.xlabel('p_B - p_A')
    plt.ylabel('Probability density')
    plt.plot(x_axis, y_axis, color='blue')
    plt.fill_between(x_axis, y_axis, color='blue', alpha=.1)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.show()

    p_A = df['p_A'][i]
    p_B = df['p_B'][i]
    Dpp = (p_B - p_A) * 100
    prob_B_higher = df['prob_B_higher'][i]

    print('Variant A performed p_A = ' + '{:.2%}'.format(p_A)
          + ' while p_B = ' + '{:.2%}'.format(p_B) + '.')
    if df['p_B'][i] > df['p_A'][i]:
        print('p_B is +' + '{:.2}'.format(Dpp) + 'pp higher than p_A.')
        print('You can be ' + '{:.0%}'.format(prob_B_higher)
              + ' confident that this is a result of the changes you made and not a result of random chance.')
    elif df['p_B'][i] < df['p_A'][i]:
        print('p_B is -' + '{:.2}'.format(-Dpp) + 'pp lower than p_A.')
        print('You can be ' + '{:.0%}'.format(1 - prob_B_higher)
              + ' confident that this is a result of the changes you made and not a result of random chance.')
    else:
        print('There is no statistical difference between variants A and B.')
