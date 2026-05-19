# NerdWallet Super Bowl 2025 Ad — Causal Inference Project

Udacity capstone analyzing the causal impact of NerdWallet's Super Bowl LIX ad
on branded Google Trends search interest.

## Files

- `nerdwallet_causal_analysis.ipynb` — executed notebook with all outputs
- `nerdwallet.csv` — Google Trends data (Nov 1, 2024 – Feb 28, 2025; US)
- `build_notebook.py` — script that builds the notebook via nbformat (not needed
  to view results, included for reproducibility)

## Methods

1. Interrupted Time Series (ITS)
2. Difference-in-Differences (DiD) — plus TWFE standout
3. Event Study — single post and time-varying daily effects
4. Synthetic Control via Lasso — α=0.1 baseline plus CV-tuned α standout

## Validation

- Parallel-trends test for DiD
- Permutation (placebo-in-space) test for SC — each donor treated as placebo
- Placebo-in-time at a single pre-event date
- Placebo-in-time sweep across 38 pre-event dates → empirical noise RMSE

## Headline result

The Super Bowl ad lifted NerdWallet branded search by **~6 Trends-index points
(~13% relative)** on average over the 19 days post-ad. The first-week spike
was much larger (~58 pts on Feb 10) and decayed to a ~2–3 pt residual lift
by Feb 28. Preferred model: Synthetic Control with CV-tuned α; inference via
permutation test (z ≈ 11.85 vs the placebo distribution).

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter nbformat nbconvert
python build_notebook.py
jupyter nbconvert --to notebook --execute nerdwallet_causal_analysis.ipynb \
    --output nerdwallet_causal_analysis.ipynb
```
