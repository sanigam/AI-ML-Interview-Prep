# Time Series Forecasting: Advanced Methods

📺 **Video Lecture:** https://youtu.be/ziH9_Ahgq4s

## Interview Anchor
- **ARIMA/SARIMA:** Autoregressive Integrated Moving Average models; statistical frameworks combining AR, differencing, and MA components
- **Modern Approaches:** Deep learning (LSTM, CNN, Transformer), Prophet, state space models; handle complex patterns and exogenous variables
- **Evaluation:** Specialized metrics (MAE, MAPE, MASE) and walk-forward backtesting ensuring realistic forecast quality assessment

## Key Concepts Overview
Advanced time series forecasting bridges classical statistical methods (ARIMA, GARCH) and modern deep learning approaches, each suited to different data regimes. The Box-Jenkins methodology provides a systematic framework for ARIMA model selection, but Prophet democratizes forecasting for practitioners lacking statistical expertise. State space models and Kalman filters offer elegant probabilistic frameworks, while neural approaches (LSTM, Temporal Fusion Transformers) capture nonlinearities and long-range dependencies classical methods miss. Understanding when to use each method, how to evaluate forecasts honestly, and techniques for handling exogenous variables and cointegration separates advanced practitioners. This section covers the complete toolkit: from parameter selection algorithms to ensemble strategies and multi-step forecasting pitfalls.

---

### Q1: Explain the Box-Jenkins methodology for ARIMA model building. What are the steps?

**A:** **Box-Jenkins** is the classical, systematic procedure for fitting an ARIMA(p, d, q) model.

**Five-step workflow:**

1. **Stationarity check.** Plot the series and run ADF / KPSS tests. If non-stationary, apply differencing (order d) until stationary. Retest after each differencing step.
2. **ACF/PACF analysis** of the differenced series:
   - PACF cuts off at lag p, ACF decays → suggests AR(p).
   - ACF cuts off at lag q, PACF decays → suggests MA(q).
   - Both decay → ARMA — use AIC/BIC to pick orders.
3. **Model fitting.** Estimate ARIMA(p, d, q) parameters via maximum likelihood.
4. **Diagnostic checking.** Plot residuals and their ACF — residuals should look like white noise (no autocorrelation, constant mean and variance). If not, iterate by increasing p or q.
5. **Forecasting.** Generate point forecasts and confidence intervals.

The methodology is iterative — ACF/PACF guidance is refined by residual diagnostics. Modern practice supplements Box-Jenkins with automated AIC/BIC selection (`auto.arima` in R, `auto_arima` in Python's `pmdarima`), but understanding the manual process matters in interviews — it shows statistical rigor and diagnostic thinking.

---

### Q2: How do you select p, d, q parameters in ARIMA(p,d,q)?

**A:** Three steps, one for each parameter.

**1. Differencing order d.** Use the ADF test iteratively:

- Start with d = 0; if ADF rejects stationarity, set d = 1 and retest.
- Most real series are I(1) — one difference achieves stationarity.
- Going beyond d = 2 is rare; over-differencing introduces artificial autocorrelation.

**2. AR order p and MA order q** — read off the ACF and PACF of the differenced series:

- Pure AR(p) — PACF has p significant spikes then cuts off; ACF decays smoothly.
- Pure MA(q) — ACF has q significant spikes then cuts off; PACF decays.
- ARMA(p, q) — both ACF and PACF decay smoothly.

**3. Tie-breaking with information criteria.** When ACF/PACF suggest multiple plausible orders, use:

```
AIC  =  2·k  −  2·ln(L)              # k = #parameters, L = likelihood
BIC  =  k·ln(n)  −  2·ln(L)           # n = sample size
```

Lower is better. Both penalize complexity; BIC penalizes more aggressively for large n. A standard procedure is grid search over p ∈ [0, 5], q ∈ [0, 5] and pick the pair minimizing AIC or BIC.

**Tooling.** `auto.arima` (R) and `auto_arima` (Python's `pmdarima`) automate this grid search. Use them as a baseline, but always sanity-check via ACF/PACF and residual diagnostics — the insight that "ACF/PACF are actionable, not decorative" is what separates competent practitioners.

---

### Q3: What is SARIMA (Seasonal ARIMA) and how do you extend ARIMA to handle seasonality?

**A:** **SARIMA** extends ARIMA with a seasonal component, written as:

```
SARIMA(p, d, q)(P, D, Q, s)
```

where:

- (p, d, q) — non-seasonal orders (same as ARIMA).
- (P, D, Q) — seasonal orders.
- s — seasonal period (e.g., s = 12 for monthly data with yearly seasonality).

The model combines ARIMA(p, d, q) on the (possibly differenced) data with a seasonal ARIMA(P, D, Q) on the seasonally differenced data:

```
seasonal differencing:  y_t − y_{t−s}
```

**Reading seasonal orders from plots.** Examine the ACF/PACF of the seasonally differenced series at lags that are multiples of s (12, 24, 36, ... for s = 12). The pattern at those seasonal lags determines (P, D, Q) the same way the pattern at lags 1, 2, 3, ... determines (p, d, q).

**Example.** SARIMA(1, 1, 1)(1, 1, 1, 12) — non-seasonal AR(1) and MA(1) with one regular difference, plus seasonal AR(1) and MA(1) with one seasonal difference and period 12.

**Recommended workflow:**

1. Determine d via ADF on the original series.
2. Determine D via ADF on the seasonally differenced series.
3. Determine p, q, P, Q via ACF/PACF.
4. Grid search the combination using AIC/BIC.

SARIMA handles complex seasonal patterns more flexibly than Holt-Winters but is more expensive to fit. In interviews, the key point is that seasonal differencing y_t − y_{t−s} is an *integral part of the model specification*, not just a preprocessing step.

---

### Q4: Explain VAR (Vector Autoregression) models for multivariate time series.

**A:** **VAR** extends autoregression to multiple time series, with each series regressed on its own lags *and* lags of every other series. For two series y_t and x_t, a VAR(p) looks like:

```
y_t  =  a₀  +  Σᵢ (a₁,ᵢ · y_{t−i}  +  a₂,ᵢ · x_{t−i})  +  ε_y,t

x_t  =  b₀  +  Σᵢ (b₁,ᵢ · y_{t−i}  +  b₂,ᵢ · x_{t−i})  +  ε_x,t
```

Each equation can be fit via OLS independently.

**Strengths:**

- Captures cross-series dependencies and bidirectional causality (basis for Granger causality tests).
- Easy to estimate (OLS per equation).
- Standard tool in macroeconomics and multi-asset finance.

**Limitations:**

- Assumes stationarity of all series — difference first if unit roots are present.
- Parameter count grows fast: K series with p lags need ~K² · p parameters.
- Impulse response functions and forecast error variance decomposition need careful, domain-informed interpretation.

**Order selection:** grid-search p with AIC/BIC.

**Granger causality:** test whether x_{t−i} terms add predictive power to the y_t equation beyond y's own lags.

**Related methods:** for *cointegrated* (non-stationary but linked) series, VECM (Vector Error Correction Model) preserves the long-run equilibrium relationship that plain VAR on differenced data would lose. In interviews, frame VAR as the workhorse for stationary multivariate forecasting, with VECM as the next step when cointegration enters the picture.

---

### Q5: What is GARCH (Generalized Autoregressive Conditional Heteroskedasticity) and when is it used?

**A:** **GARCH** models *time-varying volatility* — the conditional variance of a series. It's especially useful for financial returns, which exhibit **volatility clustering** (calm periods followed by turbulent ones).

For a return series:

```
r_t = μ + ε_t
```

GARCH(p, q) models the conditional variance σ_t² as a function of past squared shocks and past variances:

```
σ_t²  =  ω  +  Σᵢ αᵢ · ε_{t−i}²  +  Σⱼ βⱼ · σ_{t−j}²
```

The simplest and most-used variant is **GARCH(1, 1)**:

```
σ_t²  =  ω  +  α · ε_{t−1}²  +  β · σ_{t−1}²
```

So a recent shock (ε_{t−1}²) and recent variance (σ_{t−1}²) drive the current variance.

**Benefits:**

- Captures volatility clustering — a spike persists for a while.
- Produces time-varying confidence intervals around forecasts.
- Essential for risk management (Value at Risk).

**Estimation:** maximum likelihood with normal or Student-t errors.

**Variants:**

- **EGARCH** — asymmetric; negative shocks affect volatility differently than positive ones.
- **GJR-GARCH** — another asymmetric variant.
- **Multivariate GARCH** — joint volatility of multiple assets.

**Composition with ARIMA.** GARCH is orthogonal to mean modeling — you can model the mean with ARIMA(p, d, q) and the variance with GARCH(P, Q), giving an ARIMA-GARCH model. In interviews, GARCH usually only comes up if you've worked with financial data — mention it in that context.

---

### Q6: What is Prophet and what problems does it solve?

**A:** **Prophet** (originally from Facebook) is a forecasting library that automates much of time-series analysis. It models the series as the sum of three components:

- **Trend** — piecewise linear (or logistic for saturation), with automatically detected changepoints.
- **Seasonality** — additive or multiplicative, modeled as Fourier series (handles multiple periodicities like daily + weekly + yearly).
- **Holiday effects** — explicit holiday dummies for known events.

**What Prophet handles well out of the box:**

- **Missing data** — interpolates automatically.
- **Outliers** — robust loss handles them gracefully.
- **Model specification** — automatic trend changepoints and seasonality detection.
- **Non-expert users** — simple, interpretable API.

**Compared to ARIMA:**

- *Prophet:* minimal tuning, multiple seasonalities easy, holidays included naturally.
- *ARIMA/SARIMA:* requires more diagnostic work but is often more accurate when data is clean.

**Prophet's limitations:**

- Assumes specific functional forms for trend and seasonality — can miss truly complex patterns.
- Often less accurate than a well-tuned ARIMA or SARIMA on clean data.
- Uncertainty intervals can be wider than necessary.

**When to reach for which:** Prophet when data is messy, domain expertise is limited, or interpretability matters; ARIMA when data is clean and you can invest in diagnostics. In interviews, mentioning Prophet shows you understand production-grade forecasting (robustness, automation) alongside academic rigor (assumptions, diagnostics).

---

### Q7: Explain state space models and Kalman filtering for time series.

**A:** **State space models** represent a time series via a hidden state that evolves over time. They consist of two equations.

**Observation equation** — what you actually see, given the hidden state:

```
y_t = H · x_t + v_t,    v_t ~ Normal(0, R)
```

**State (transition) equation** — how the hidden state evolves:

```
x_t = F · x_{t−1} + w_t,    w_t ~ Normal(0, Q)
```

**The Kalman filter** is a recursive algorithm that estimates the hidden state given observations, minimizing mean squared error. Each step has two phases:

1. **Predict** — estimate state and covariance at time t given history up to t − 1.
2. **Update** — incorporate the new observation y_t and refine the estimate.

The filter runs in O(n) time and gracefully handles missing observations (just skip the update step when y_t is missing).

**Benefits:**

- Time-varying estimates of unobserved components (trend, seasonal, level).
- Uncertainty quantification via posterior covariance.
- Native handling of missing and irregularly spaced data.

**Connection to other models.** ARIMA and exponential smoothing are special cases of state space models. The "local linear trend" state-space model (level + trend states) is equivalent to Holt's exponential smoothing.

In interviews, state space models signal theoretical sophistication — mention them if you've worked on decomposition or missing-data problems. The Kalman filter is also the gold standard in aerospace and sensor fusion (GPS tracking, navigation).

---

### Q8: How do LSTM and neural networks approach time series forecasting? What are advantages and disadvantages?

**A:** **LSTMs** are recurrent neural networks with gating mechanisms (forget, input, output gates) that capture long-range dependencies in sequences. For forecasting, an LSTM consumes a sequence of length L and predicts the next value(s) — no hand-crafted features needed.

**Architectural variants:**

- **Stacked LSTMs** — multiple LSTM layers for richer temporal abstractions.
- **Bidirectional LSTMs** — encode the sequence both forwards and backwards (training-time only — bidirectional doesn't make sense for online forecasting).
- **Encoder-decoder with attention** — the seq2seq pattern; encoder summarizes the history, decoder generates multi-step forecasts attending to encoder outputs.

**Advantages:**

- No assumptions about stationarity or linearity.
- Captures complex nonlinear patterns.
- Naturally handles variable-length sequences and exogenous covariates (just concatenate them at each timestep).

**Disadvantages:**

- Data-hungry — needs thousands of training samples to avoid overfitting.
- Expensive hyperparameter tuning (architecture, learning rate, dropout, batch size).
- Black-box — hard to diagnose why a forecast went wrong.
- Slower inference than ARIMA / Prophet.
- Training can be unstable; gradient clipping and learning-rate scheduling are usually necessary.

**When to use:** large datasets where standard methods fail and the nonlinearity is real. Always start with ARIMA / Prophet baselines and only escalate to LSTMs if the gap is meaningful.

In interviews, LSTM forecasting signals cutting-edge knowledge — but being honest about data and compute requirements impresses more than overselling.

---

### Q9: What is the Temporal Fusion Transformer (TFT) and how does it extend transformer attention to time series?

**A:** **TFT (Temporal Fusion Transformer)** is a transformer-based architecture for multivariate forecasting that combines temporal convolutions, attention, and learned variable selection.

**Key innovations:**

- **Variable selection networks** — learn per-timestep importance weights for input features, providing built-in explainability.
- **Temporal self-attention** — attend to relevant past timesteps without a fixed receptive field like convolutions.
- **Quantile regression** — predict quantiles (e.g., 0.1, 0.5, 0.9) rather than point forecasts, giving calibrated uncertainty intervals.
- **Multi-horizon forecasting** — predict multiple steps ahead jointly in one forward pass.

**Architecture sketch.** Embed inputs, pass through variable selection, apply temporal convolutions for local patterns, then multi-head self-attention for global temporal context, feed-forward layers, and finally a quantile decoder.

**Advantages:**

- Interpretable — attention weights and variable-selection weights are inspectable.
- Handles mixed data — embeddings for categorical features, normalization for continuous.
- Produces calibrated uncertainty estimates via quantile loss.

**Disadvantages:**

- Complex architecture with many hyperparameters.
- Data-hungry.
- Slower to train than LSTMs.

**Production status.** TFT is strong on medium-horizon forecasting benchmarks but less battle-tested than ARIMA/Prophet/LSTM in industry production. In interviews, mentioning TFT shows awareness of recent advances — but pair it with realism about cost-benefit tradeoffs in real systems.

---

### Q10: What is ensemble forecasting and why is it effective?

**A:** **Ensemble forecasting** combines predictions from multiple models to reduce individual model errors.

**Common methods:**

- **Simple averaging** — average the forecasts of several models:

  ```
  forecast_ensemble = (forecast_ARIMA + forecast_ExpSmooth + forecast_Prophet) / 3
  ```

  Reduces variance without bias if the component models are diverse and roughly unbiased.

- **Weighted averaging** — assign higher weights to more accurate models, with weights determined by validation performance.

- **Stacking** — train a meta-learner (e.g., linear regression) on the base models' predictions to learn the optimal combination.

- **Boosting** — sequentially train models that correct previous errors. Less common in time-series forecasting.

**Why ensembles work:**

- More stable than a single model.
- Often more accurate than the best individual component.
- Robust to model-specific failure modes (if ARIMA underfits the nonlinearity, Prophet may catch it).

**Tradeoff:** more compute and operational complexity.

**Best practice.** Ensemble diverse models — pair a statistical method (ARIMA) with a learning-based one (LSTM or gradient boosting), and tune weights on a validation set. In forecasting competitions, ensembles are nearly mandatory; in production, weigh the latency cost against the accuracy gain.

In interviews, ensemble thinking signals maturity — individual models have specific strengths and weaknesses, and combining them hedges against any one failing.

---

### Q11: Explain multi-step ahead forecasting. What's the difference between recursive and direct methods?

**A:** **Multi-step (multi-horizon) forecasting** predicts multiple steps into the future:

```
ŷ_{t+1}, ŷ_{t+2}, ..., ŷ_{t+h}
```

where h is the horizon. There are two main strategies.

**Recursive (iterated).** Train a single-step model and apply it iteratively — feed each prediction back as the next step's input:

```
predict ŷ_{t+1}  from  y_t
predict ŷ_{t+2}  from  ŷ_{t+1}      (using the predicted value)
...
predict ŷ_{t+h}  from  ŷ_{t+h−1}
```

- *Pros:* one model, simple, computationally efficient.
- *Cons:* errors accumulate — each step adds bias and variance from feeding predicted (rather than true) values back in.

**Direct.** Train a separate model for each horizon — one to predict ŷ_{t+1}, another to predict ŷ_{t+2} directly from y_t, and so on.

- *Pros:* no compound error, each model targets its specific horizon.
- *Cons:* h separate models, less data per model, more engineering effort.

**Hybrid (DirRec, etc.)** — combine both ideas, training rolling-window models that use a mix of true and predicted past values to balance error compounding and per-horizon bias.

**Rules of thumb:**

- Short horizons (h ≤ 5) → recursive is usually fine.
- Long horizons → direct or hybrid; consider training-set augmentation.
- In practice, recursive + ensemble (multiple recursive models combined) often gives the best balance and provides natural uncertainty estimates.

In interviews, knowing the recursive vs direct tradeoff signals you've dealt with real multi-step forecasting in production.

---

### Q12: What are common time series forecasting evaluation metrics? When do you use each?

**A:** A quick tour of the standard forecasting metrics.

**MAE — Mean Absolute Error:**

```
MAE = (1/n) · Σ_t | y_t − ŷ_t |
```

Robust to outliers, interpretable in original units. Preferred for business metrics ("forecast error in dollars or units").

**RMSE — Root Mean Squared Error:**

```
RMSE = √( (1/n) · Σ_t (y_t − ŷ_t)² )
```

Penalizes large errors more heavily; sensitive to outliers. Preferred when big misses are particularly costly.

**MAPE — Mean Absolute Percentage Error:**

```
MAPE = (1/n) · Σ_t  | y_t − ŷ_t |  /  | y_t |
```

Scale-invariant — useful when comparing across series of different magnitudes. Caution: undefined when y_t = 0 and inflates wildly when y_t is small.

**SMAPE — Symmetric MAPE:**

```
SMAPE = (1/n) · Σ_t  2 · | y_t − ŷ_t |  /  ( | y_t | + | ŷ_t | )
```

Bounded in [0, 2]; addresses some of MAPE's issues with small values.

**MASE — Mean Absolute Scaled Error:**

```
MASE = MAE / MAE_naive_baseline
```

Scale-invariant benchmark against a naive baseline (e.g., ŷ_t = y_{t−1}). MASE = 1 matches the baseline; < 1 beats it.

**Directional accuracy** — percentage of times the forecast's direction (up/down) matches the actual. Critical for trading decisions where being on the right side matters more than the magnitude.

**Choice cheat sheet:**

- *Default for business metrics:* MAE.
- *Outliers especially costly:* RMSE.
- *Comparing across scales:* MAPE or SMAPE — but skip MAPE for small/zero-heavy values.
- *Scale-invariant benchmarking:* MASE.
- *Trading / sign-sensitive decisions:* directional accuracy + Sharpe ratio.

Always report multiple metrics — single metrics can mislead. In interviews, mentioning MASE signals sophistication; MAPE familiarity is expected.

---

### Q13: What is backtesting in time series forecasting? How do you design a credible backtesting strategy?

**A:** **Backtesting** evaluates a forecasting model on historical data, simulating real-world deployment. The fundamental rule: **never use future data during training** — respect temporal order strictly.

**Walk-forward backtesting** (expanding window):

1. Train on [t₁ ... t₆₀]; forecast [t₆₁ ... t₆₅] (5 steps ahead).
2. Evaluate forecasts against actual [t₆₁ ... t₆₅].
3. Expand the window: train on [t₁ ... t₆₅]; forecast [t₆₆ ... t₇₀].
4. Repeat until the end of the dataset, accumulating metrics over all test windows.

A **rolling-window** variant keeps the window size fixed (drops oldest observations) — useful when older data isn't representative.

**Pitfalls to avoid:**

- **Overlapping test windows** — leakage between consecutive evaluations.
- **Overfitting to the backtest period** — tune hyperparameters on a separate validation set, not on backtest results.
- **Survivorship bias** — silently dropping observations that wouldn't be available in real time.
- **Regime changes** — historical patterns may not predict future behavior; check robustness across time periods.

**Standard split.** Train (~70%) → validation (~10%) → test (~20%), respecting time order. Tune on validation, report final performance on test.

In interviews, proper backtesting is critical — naive practitioners frequently report inflated accuracy due to leakage. Mention walk-forward explicitly, talk about information leakage, and bring up regime changes if relevant to the problem.

---

### Q14: What are exogenous variables in forecasting? How do you handle them?

**A:** **Exogenous variables** are external predictors that aren't the target series but help explain it — temperature for electricity demand, holidays and promotions for sales, oil prices for airline revenue.

**Models that incorporate exogenous variables:**

- **ARIMAX (ARIMA-X)** — ARIMA with an exogenous regression term added to the mean.
- **Dynamic regression** — regression with ARIMA errors. The mean is modeled by the exogenous variables; the residuals follow an ARIMA process.
- **VAR** — joint model of multiple series, capturing temporal *and* cross-series dependencies.
- **Neural networks** — feed exogenous covariates alongside the target sequence; LSTMs and TFTs handle this naturally.

**Practical challenges:**

- **Future availability.** Forecasting electricity demand needs the *future* temperature — usually requires a separate temperature forecast.
- **Leakage from endogenous variables.** If a "covariate" is actually influenced by the target (bidirectional causality), regression coefficients become biased.
- **Feature selection.** Too many exogenous variables inflate variance; use domain knowledge or Granger causality tests to prune.
- **Missing data.** Impute or forecast separately.

**Best practice.** Validate that exogenous variables really are exogenous (Granger causality test) and that they'll be available at forecast time. In interviews, demonstrating that you think about *availability* and *leakage* is impressive — most naive practitioners only think about whether the variable correlates with the target.

---

### Q15: Explain cointegration and Granger causality in multivariate time series.

**A:** Two key concepts in multivariate time series.

**Cointegration.** Two non-stationary series are **cointegrated** if some linear combination of them is stationary. The intuition: even if the individual series wander (unit roots), they share a long-run equilibrium that prevents them from drifting apart indefinitely.

**Example.** Two stock prices that are both I(1) (non-stationary) may have a stationary *spread*, because they're driven by the same underlying factors. Pairs trading exploits exactly this — when the spread deviates from its mean, mean-reversion creates a trade.

**Engle-Granger test for cointegration:**

1. Regress y on x.
2. Test whether the regression residuals are stationary (e.g., ADF on residuals).
3. If residuals are stationary even though y and x are not, they're cointegrated.

**VECM (Vector Error Correction Model)** is the right tool for cointegrated systems — it captures both short-term dynamics and the long-term equilibrium. Plain VAR on *differenced* data would discard the cointegration information, which is why VECM matters.

---

**Granger causality.** Series x **Granger-causes** y if past values of x improve the forecast of y beyond what y's own history alone can do.

**The test:**

1. Regress y on its own lags.
2. Regress y on its own lags *plus* lags of x.
3. Test whether the x-lag coefficients are jointly significant. If so, x Granger-causes y.

**Critical caveat.** Granger causality is **predictive** causality — it does *not* establish true causality.

**Common counterexample.** A common driver z affects both x and y, with x affecting y first (or being measured first). Then x Granger-causes y, but the *real* cause is z. Use Granger tests as exploratory tools, not as proof of causal mechanism.

In interviews, cointegration and Granger causality together show sophistication in multivariate forecasting; mentioning VECM signals theoretical depth. The cautionary note about Granger ≠ true causation is what really lands.

---

## Interview Cheatsheet

**Key Terms:**
- **Box-Jenkins Methodology:** Systematic ARIMA(p,d,q) selection via stationarity testing, ACF/PACF analysis, and residual diagnostics
- **SARIMA:** Seasonal ARIMA extending ARIMA to include seasonal components (P,D,Q) and seasonal period s
- **VAR (Vector Autoregression):** Multivariate AR model capturing cross-series dependencies; requires stationarity
- **GARCH:** Models time-varying volatility via autoregressive conditional heteroskedasticity; financial applications
- **Prophet:** Automated forecasting library robust to missing data, outliers, multiple seasonalities; Facebook-developed
- **State Space Model:** Hidden state evolving per state equation, observations related via observation equation; Kalman filter estimates states
- **LSTM:** Recurrent neural network with gating mechanisms for long-range dependencies; deep learning approach to forecasting
- **Temporal Fusion Transformer (TFT):** Transformer-based architecture with variable selection, temporal attention, quantile regression
- **Ensemble Forecasting:** Combines multiple model forecasts (averaging, weighting, stacking) for improved stability and accuracy
- **Multi-step Forecasting:** Predicting multiple steps ahead; recursive (iterated) vs. direct methods; compound error risk in recursive
- **Walk-Forward Backtesting:** Sequential expanding window evaluation respecting temporal order; prevents information leakage
- **MASE:** Mean Absolute Scaled Error; scale-invariant metric comparing to naive baseline; preferred for benchmarking
- **Exogenous Variables:** External predictors improving forecast; availability and true exogeneity are critical
- **Cointegration:** Linear combination of non-stationary series is stationary; implies long-term equilibrium relationship
- **Granger Causality:** x Granger-causes y if x's past improves y's forecast; predictive, not true causality
- **VECM:** Vector Error Correction Model; handles cointegrated systems capturing equilibrium and short-term dynamics

**Rapid-Fire Q&A:**
- **Q: How to choose ARIMA order (p,d,q)?** **A:** ADF test for d; ACF/PACF for p,q; grid search with AIC/BIC
- **Q: When does ARIMA fail?** **A:** Nonlinear relationships, multiple seasonalities, missing exogenous variables, regime shifts
- **Q: Prophet vs. ARIMA?** **A:** Prophet more robust to messiness, less tuning; ARIMA more accurate if clean, time-intensive
- **Q: LSTM vs. statistical methods?** **A:** LSTM for big data/complex patterns; statistical for small data, interpretability
- **Q: Recursive or direct for multi-step?** **A:** Recursive efficient, compounds error; direct slower but no compound error; hybrid often best
- **Q: How to compare forecast models?** **A:** Walk-forward backtest, multiple metrics (MAE, RMSE, MASE), ensemble if uncertain
- **Q: Cointegration vs. correlation?** **A:** Cointegration: linear combo is stationary; correlation: simple dependence
- **Q: What's Granger causality weakness?** **A:** Predictive, not true causality; reverse causality and common causes confound
- **Q: How to forecast with missing exog data?** **A:** Forecast exogenous separately, or condition on available scenarios
- **Q: Evaluation metric for trading?** **A:** Directional accuracy (up/down calls); Sharpe ratio if returns

---

## Interview Tips
- **Start with baselines:** Always report ARIMA/Prophet/naive forecast before complex methods; compare against them
- **Diagnose vs. predict:** Deep understanding of why a model works (stationarity assumptions, parameter meanings) impresses more than accuracy alone
- **Discuss computational trade-offs:** LSTM is glamorous but slower; mention if production latency constraints favor statistical methods
- **Backtesting rigor is everything:** Incorrectly designed backtesting is worse than no backtesting; walk-forward is non-negotiable
- **Prepare multivariate example:** Show you've handled multiple series with VAR or cointegration; real forecasting often multivariate
- **Mention modern tools:** Darts (Python library combining ARIMA, Prophet, NN), Nixtla's StatsForecast (automated), AutoML frameworks
- **Be honest about forecast limits:** Forecasts are uncertain; confidence intervals and scenario analysis matter as much as point forecasts
- **Connect to business metrics:** Translate forecast accuracy (MASE) to business impact (dollars saved, decisions improved)
- **Prepare for ensemble discussion:** Show you understand why ensembles work; pick two contrasting methods and explain complementarity

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*
