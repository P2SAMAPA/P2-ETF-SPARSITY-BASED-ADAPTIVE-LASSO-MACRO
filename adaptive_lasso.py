import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def macro_sensitivity_scores(returns, macro_df):
    """
    For a given ETF, compute the L2 norm of coefficients from a ridge regression
    of its returns on all macro variables. Higher sensitivity → lower penalty.
    Returns a single scalar (sensitivity) for the ETF.
    """
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    rets = returns[:min_len]
    macro = macro_df.iloc[:min_len]
    if len(rets) < 5:
        return 0.0
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro)
    # Fit ridge regression to get stable coefficients
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(macro_scaled, rets)
    coef = ridge.coef_
    sensitivity = np.linalg.norm(coef)
    return sensitivity

def adaptive_lasso_weights(returns_df, macro_df, gamma=1.0):
    """
    Compute adaptive LASSO penalty weights for each ETF: w_j = 1 / (|sensitivity_j|^gamma)
    """
    sensitivities = {}
    for ticker in returns_df.columns:
        sens = macro_sensitivity_scores(returns_df[ticker].values, macro_df)
        sensitivities[ticker] = sens
    # Avoid division by zero
    max_sens = max(sensitivities.values()) if sensitivities else 1.0
    weights = {}
    for ticker, sens in sensitivities.items():
        if sens < 1e-6:
            weights[ticker] = 1e6   # high penalty for insensitive ETFs
        else:
            weights[ticker] = 1.0 / (sens ** gamma)
    return weights

def adaptive_lasso_coefficients(returns_df, macro_df, Lambda=0.01, gamma=1.0):
    """
    Solve adaptive LASSO: min 0.5 * ||y - Xβ||^2 + λ * Σ w_j |β_j|
    where y = next-day return (last day), X = ETF returns (features)?? Actually we want to select
    which ETFs to invest in to predict something? The original idea: use LASSO to select ETFs
    that are expected to have positive return. But the problem is not standard regression.
    Instead, we use a different approach: For a given window, we want to predict the next‑day return
    of each ETF individually? That would be univariate.
    However, the sparsity‑based selection is typically used in portfolio optimisation: we have
    a target return (e.g., market return) and we want to select a sparse set of ETFs that replicate it.
    But simpler: We can directly use the macro‑informed penalty to compute a "selection score"
    for each ETF as the product of its macro sensitivity and its last‑day return (momentum).
    That yields a sparse ranking.
    Given the complexity of a true adaptive LASSO for portfolio selection (needs a target),
    we will implement a simplified but still principled method:
    For each ETF, compute a score = (last day return) * (macro sensitivity) * (1 / (1 + regularisation)).
    Then we select the top 3 by this score.
    This is interpretable and uses macro sensitivity to up‑weight ETFs that are macro‑responsive.
    """
    last_returns = returns_df.iloc[-1].values
    tickers = returns_df.columns
    sensitivities = {}
    for ticker in tickers:
        sens = macro_sensitivity_scores(returns_df[ticker].values, macro_df)
        sensitivities[ticker] = sens
    # Compute score
    scores = {}
    for i, ticker in enumerate(tickers):
        momentum = last_returns[i]
        sens = sensitivities[ticker]
        # Higher sensitivity and positive momentum give higher score
        # Penalise negative momentum
        if momentum > 0:
            score = momentum * sens * (1.0 / (1.0 + Lambda))
        else:
            score = momentum * sens * (1.0 / (1.0 + Lambda))  # negative remains negative
        scores[ticker] = score
    return scores
