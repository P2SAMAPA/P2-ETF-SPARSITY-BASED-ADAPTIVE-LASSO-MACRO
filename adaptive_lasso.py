import numpy as np
from sklearn.linear_model import Ridge
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
    # Remove any rows where returns or macro have NaN
    mask = np.isfinite(rets)
    if not np.all(mask):
        rets = rets[mask]
        macro = macro.iloc[mask]
    if len(rets) < 5:
        return 0.0
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro)
    # Fit ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(macro_scaled, rets)
    coef = ridge.coef_
    sensitivity = np.linalg.norm(coef)
    return sensitivity

def adaptive_lasso_coefficients(returns_df, macro_df, Lambda=0.01, gamma=1.0):
    """
    Compute a selection score for each ETF: last return * macro sensitivity * (1/(1+λ)).
    """
    last_returns = returns_df.iloc[-1].values
    tickers = returns_df.columns
    sensitivities = {}
    for ticker in tickers:
        # Get returns series for this ETF
        ret_series = returns_df[ticker].values
        # Align with macro (same index)
        # Ensure both have same length and no NaNs
        min_len = min(len(ret_series), len(macro_df))
        rets = ret_series[:min_len]
        macro = macro_df.iloc[:min_len]
        # Remove rows with NaN in rets
        mask = np.isfinite(rets)
        if not np.all(mask):
            rets = rets[mask]
            macro = macro.iloc[mask]
        if len(rets) < 5:
            sens = 0.0
        else:
            sens = macro_sensitivity_scores(rets, macro)
        sensitivities[ticker] = sens
    # Compute score
    scores = {}
    for i, ticker in enumerate(tickers):
        momentum = last_returns[i]
        sens = sensitivities[ticker]
        # Higher sensitivity and positive momentum give higher score
        if momentum > 0:
            score = momentum * sens * (1.0 / (1.0 + Lambda))
        else:
            score = momentum * sens * (1.0 / (1.0 + Lambda))  # negative remains negative
        scores[ticker] = score
    return scores
