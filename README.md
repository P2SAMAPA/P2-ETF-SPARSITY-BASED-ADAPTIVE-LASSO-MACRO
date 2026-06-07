# Adaptive LASSO with Macro-Informed Penalty for ETFs

Implements a sparsity‑based ETF selection method using adaptive LASSO. Penalty weights are derived from ETF sensitivity to macro variables (VIX, DXY, yields): more macro‑sensitive ETFs receive lower penalty and are more likely to be selected. The per‑ETF score combines last‑day return (momentum) and macro sensitivity.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Uses all available macro variables to compute sensitivity (ridge regression L2 norm)
- Adaptive penalty: w_j = 1 / (sensitivity^γ)
- Score = last‑day return × sensitivity × (1/(1+λ))
- Positive score → candidate for long; negative → avoid
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-adaptive-lasso-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, O(n²) per window)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High positive score → ETF has positive momentum and is highly responsive to macro → likely to continue upward.
- Low or negative score → avoid.

## Requirements

See `requirements.txt`.
