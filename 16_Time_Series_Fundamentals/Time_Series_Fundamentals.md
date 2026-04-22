# Time Series Fundamentals

📺 **Video Lecture:** https://youtu.be/YqHagchJ20Y


## Interview Anchor
- **Stationarity:** Time-invariant statistical properties (mean, variance, autocorrelation); essential for classical time series methods
- **Trend and Seasonality:** Deterministic components capturing long-term movement and regular periodic patterns
- **Autocorrelation:** Correlation of a series with its past values; key diagnostic for model selection and dependencies

## Key Concepts Overview
Time series analysis is critical in finance, forecasting, and sensor data applications, and interviewers expect deep understanding of how time dependencies violate iid assumptions. Unlike standard machine learning, time series data has temporal structure—past observations influence future ones—requiring specialized handling. Stationarity is the foundation: many classical methods (ARIMA, GARCH) assume stationary data, so practitioners must diagnose stationarity and apply transformations (differencing, log transforms) to achieve it. Understanding autocorrelation (ACF/PACF plots), seasonal decomposition, and how to handle missing values in temporal sequences separates competent practitioners from novices. This section covers essential diagnostics and preprocessing techniques that enable downstream forecasting success.

---

### Q1: What is stationarity in time series? Distinguish between strict and weak stationarity.

**A:** Stationarity means the statistical properties of a time series don't change over time. Strict stationarity requires the joint distribution of (yₜ, yₜ₊₁, ..., yₜ₊ₙ) to be identical for all shifts in time t, which is very restrictive. Weak (or covariance) stationarity only requires constant mean E[yₜ] = μ, constant variance Var(yₜ) = σ², and autocovariance Cov(yₜ, yₜ₊ₖ) depending only on lag k (not on t). In practice, weak stationarity is sufficient for most classical methods like ARIMA. A non-stationary series exhibits a trend (changing mean), seasonal patterns (time-varying variance or periodicity), or unit roots (random walks). Classic example: stock prices are non-stationary (have trends), but returns (price changes) are approximately stationary. Many forecasting models assume stationarity, so detecting and transforming non-stationary data is critical before modeling.

---

### Q2: What is a trend in time series? How do you detect and remove it?

**A:** A trend is a long-term, persistent movement in the series' level—an upward or downward drift. Trends violate stationarity by changing the mean over time. You detect trends visually (plotting the series) or statistically via regression against time t or trend tests (Augmented Dickey-Fuller test rejects stationarity if trend is strong). Removal methods include: (1) Differencing: compute yₜ' = yₜ - yₜ₋₁; first-order differencing removes linear trends, higher-order removes polynomial trends. (2) Detrending: fit a trend model (e.g., polynomial regression yₜ = a + bt + εₜ) and subtract the fitted trend. (3) Log transformation: for exponential trends, log(yₜ) linearizes exponential growth. Differencing is preferred in ARIMA frameworks because it's invertible (forecasts can be reverse-transformed) and integrates into the model naturally as the I(d) component. Always remove trends before assuming stationarity; otherwise, statistical tests and confidence intervals are invalid.

---

### Q3: Explain seasonality. How is it different from a trend?

**A:** Seasonality is a regular, periodic pattern repeating at fixed intervals—daily, weekly, monthly, or yearly cycles. Unlike trends (which drift monotonically), seasonality oscillates around a level and resets predictably. Example: retail sales spike every December, temperature has daily cycles, web traffic peaks on weekdays. Seasonality can be additive (seasonal pattern is constant magnitude: yₜ = Tₜ + Sₜ + εₜ) or multiplicative (seasonal strength scales with level: yₜ = Tₜ × Sₜ × εₜ). Additive fits when seasonal swings are roughly constant; multiplicative when they grow with the trend level. Detecting seasonality: (1) Visual inspection via seasonal subseries plots (plot data grouped by season). (2) Autocorrelation (ACF) plot shows spikes at seasonal lags (e.g., lag 12 for monthly data with yearly seasonality). (3) Seasonal decomposition (STL, X-11) explicitly separates trend, seasonal, and residual components. Unlike trends (removed via differencing), seasonality often requires seasonal differencing (yₜ - yₜ₋ₛ where s is the seasonal period) in ARIMA.

---

### Q4: What is autocorrelation (ACF) and partial autocorrelation (PACF)? How do you interpret them?

**A:** Autocorrelation (ACF) measures correlation between yₜ and yₜ₋ₖ (its past at lag k): ACF(k) = Cov(yₜ, yₜ₋ₖ) / Var(yₜ). Partial autocorrelation (PACF) is correlation at lag k after removing intermediate lags' effects, capturing direct dependence. ACF and PACF plots are essential diagnostics for ARIMA model selection. For an AR(p) process, PACF cuts off after lag p (non-zero up to p, then zero), while ACF decays gradually. For an MA(q) process, ACF cuts off after lag q, while PACF decays. These patterns guide choosing p and q in ARIMA(p,d,q). Significant ACF at seasonal lags (e.g., lag 12) indicates seasonality requiring seasonal ARIMA. A slow ACF decay suggests non-stationarity—confirm with ADF test. In interviews, the key insight is that ACF/PACF patterns directly prescribe ARIMA orders; they're not just fancy plots but actionable diagnostics driving model selection.

---

### Q5: What is white noise? Why is it important in time series modeling?

**A:** White noise is a sequence of independent, identically distributed random variables with mean 0 and constant variance σ². Formally, εₜ ~ WN(0, σ²) means E[εₜ] = 0, Var(εₜ) = σ², and Cov(εₜ, εₛ) = 0 for t ≠ s. An ACF plot of white noise shows no significant correlations (all within confidence bands). White noise is important because: (1) It's the target residual for time series models—if residuals are white noise, the model has captured all predictable structure. (2) Many forecasts assume errors are white noise, enabling confidence interval construction. (3) Non-white-noise residuals indicate model misspecification (missing terms, wrong order). Diagnostic check: plot residuals and their ACF; if residuals appear to be white noise (no visual pattern, no significant autocorrelation), the model is likely adequate. If residuals show autocorrelation (significant spikes in ACF), the model is underfitting and needs adjustment.

---

### Q6: Explain a random walk. Why are random walks non-stationary?

**A:** A random walk is defined as yₜ = yₜ₋₁ + εₜ where εₜ ~ WN(0, σ²). The current value equals previous value plus random shock. Examples: stock prices (efficient market hypothesis assumes prices follow random walks), cumulative sum of shocks. Random walks are non-stationary because: (1) Variance grows over time: Var(yₜ) = t·σ², not constant. (2) Mean is time-dependent if the series drifts. (3) Covariance Cov(yₜ, yₜ₋ₖ) depends on t, not just lag k. The series exhibits permanent shocks—a random walk never reverts to a mean. First differencing converts a random walk to white noise: Δyₜ = yₜ - yₜ₋₁ = εₜ, which is stationary. A random walk with drift yₜ = c + yₜ₋₁ + εₜ has a deterministic trend and strong non-stationarity; differencing removes the drift term. Detecting random walk behavior: the ACF decays very slowly (nearly 1 at all lags) and the ADF test fails to reject a unit root. Always test for unit roots before model selection.

---

### Q7: What is differencing and how does it achieve stationarity?

**A:** Differencing computes yₜ' = yₜ - yₜ₋₁, the first-order differences (changes between consecutive observations). First differencing removes linear trends and transforms I(1) series (unit root, like random walk) to I(0) stationary. Second differencing (double differencing, Δ²yₜ = Δyₜ - Δyₜ₋₁) removes quadratic trends and transforms I(2) series. Seasonal differencing (yₜ - yₜ₋ₛ where s is season length) removes seasonal patterns. Overdifferencing (applying more differencing than needed) introduces artificial autocorrelation and inflates variance; you want minimal differencing to achieve stationarity. In ARIMA(p,d,q), the d parameter specifies differencing order. Best practice: apply ADF test iteratively—if the series has unit root, difference once and retest; repeat until stationarity is achieved. Differencing is invertible (you can recover original series), essential for producing forecasts in original scale. Differencing is preferable to detrending in ARIMA because it's automatic and integrates into the model.

---

### Q8: Explain the Augmented Dickey-Fuller (ADF) test. What does it test and how do you interpret results?

**A:** The ADF test is a statistical test for a unit root (non-stationarity) in a time series. The null hypothesis is H₀: the series has a unit root (non-stationary); the alternative is H₁: the series is stationary. The test regresses Δyₜ = α + βyₜ₋₁ + Σγᵢ·Δyₜ₋ᵢ + εₜ and tests if β = 0 (equivalently, if the coefficient of yₜ₋₁ is zero, indicating a unit root). The test statistic follows a non-standard distribution (Dickey-Fuller distribution). Interpretation: (1) If p-value < 0.05 (or test statistic more negative than critical value), reject the unit root hypothesis; the series is stationary. (2) If p-value > 0.05, fail to reject; the series is non-stationary and likely needs differencing. The ADF test assumes lags of differences are included to account for autocorrelation. Best practice: apply ADF test after differencing to confirm stationarity achieved. A failing ADF before differencing and passing ADF after confirms that differencing successfully removed the unit root. This is the gold standard diagnostic before ARIMA fitting.

---

### Q9: What is the KPSS test and how does it differ from the ADF test?

**A:** KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test also tests stationarity but with opposite null hypothesis to ADF. KPSS null hypothesis H₀: the series is stationary; alternative H₁: series has a unit root (non-stationary). The test decomposes the series as yₜ = trend + random walk + error and tests if the random walk variance is zero. Interpretation: (1) If KPSS p-value < 0.05, reject stationarity (series is non-stationary). (2) If KPSS p-value > 0.05, fail to reject stationarity (series is stationary). ADF and KPSS have opposite null hypotheses, so their results can conflict (neither rejects stationarity or both reject). Best practice: apply both tests together. If ADF rejects unit root and KPSS fails to reject stationarity, you're confident the series is stationary. If ADF fails to reject unit root and KPSS rejects stationarity, the series is near the boundary (often called "nearly stationary" or "trend-stationary"), and differencing is recommended. KPSS is more powerful at detecting near-unit-root cases where ADF is inconclusive. Using both tests provides diagnostic robustness; pure KPSS or pure ADF can mislead.

---

### Q10: Explain time series decomposition. What are additive vs. multiplicative models?

**A:** Decomposition separates a time series into trend (Tₜ), seasonal (Sₜ), and residual (εₜ) components. Additive decomposition: yₜ = Tₜ + Sₜ + εₜ assumes seasonal fluctuations are constant in magnitude, independent of trend level. Multiplicative decomposition: yₜ = Tₜ × Sₜ × εₜ assumes seasonal variation scales with the trend; used when seasonal strength grows as the level increases. Example: a company's monthly sales with additive seasonality might see constant ±$1M swings around trend; multiplicative seasonality would show $1M swings when trend is $10M but $10M swings when trend is $100M. Methods to decompose: (1) Classical decomposition: compute trend via centered moving average, extract seasonal component by averaging detrended values for each season, residuals = original - trend - seasonal. (2) STL (Seasonal and Trend decomposition using LOESS): more robust, handles non-stationary trends, works for both additive and multiplicative. Choosing additive vs. multiplicative: if variance in seasonal subseries plots appears constant across seasons, use additive; if variance increases with level, use multiplicative. Log transformation (yₜ'=log(yₜ)) converts multiplicative to additive, then inverse-transform forecasts.

---

### Q11: What are moving averages and exponential smoothing in time series?

**A:** A moving average (MA) smooths a series by averaging the current value and k previous values: SMA_k(t) = (yₜ + yₜ₋₁ + ... + yₜ₋ₖ₊₁) / k. It removes noise and reveals trends but lags behind actual values. Exponential smoothing (ES) weights recent observations more heavily via SES(t) = α·yₜ + (1-α)·SES(t-1), where α ∈ (0,1) is a smoothing parameter. Higher α weights current data more (reactive, noisy); lower α gives heavier weight to history (smooth but lagging). ES adapts to level changes; MA has a fixed lag. Moving averages are used for visualization and denoising; exponential smoothing is a simple forecasting method. Holt's method extends ES to handle trends: level equation, trend equation, and forecasts. Holt-Winters extends further to include seasonality: additive or multiplicative seasonal component. All ES variants are special cases of state space models and underpin modern forecasting (Prophet, TBATS). In interviews, the key is understanding that ES methods are computationally efficient forecasting algorithms that capture level, trend, and seasonality without full ARIMA machinery.

---

### Q12: How do you construct lag features for time series regression models?

**A:** Lag features are previous values of the target or other variables used as predictors. For target yₜ, construct lag features: lag_1 = yₜ₋₁, lag_2 = yₜ₋₂, ..., lag_p = yₜ₋ₚ. These turn the time series prediction problem into supervised learning. In sliding window fashion, create (X, y) pairs: X = [yₜ₋ₚ, ..., yₜ₋₁] and y = yₜ. Lag selection: use ACF/PACF to identify significant lags (e.g., if PACF cuts off at lag 5, use p=5 lags). Other features include rolling statistics (rolling mean, rolling std), seasonal lags (yₜ₋₁₂ for monthly data with yearly seasonality), and lead-lag relationships with exogenous variables. Caution: include only past information (no future leakage). Window size (p) balances capturing temporal dependencies vs. reducing available training samples. For deep learning (LSTM, CNN), lag construction is implicit (model learns temporal dependencies); for tree models, explicit lag engineering is critical. Best practice: start with p from ACF/PACF, then validate via cross-validation and ablation studies. Lag feature engineering is less art and more science when guided by autocorrelation diagnostics.

---

### Q13: What are rolling statistics and why are they useful for time series?

**A:** Rolling statistics compute statistics over a sliding window of fixed size. Rolling mean (rolling average) smooths the series and tracks local level; rolling standard deviation captures volatility changes. Example: 30-day rolling volatility for stock returns shows when markets are calm vs. turbulent. Rolling statistics are useful for: (1) Visualization: reveal trends and volatility clusters that raw series may obscure. (2) Feature engineering: rolling statistics as predictors for downstream models (rolling mean, rolling skew, rolling correlation). (3) Anomaly detection: values deviating far from rolling mean are anomalies. (4) Seasonality detection: seasonal subseries plots are rolling means grouped by season. Exponentially weighted rolling statistics (ewm) weight recent values more (alternative to fixed-window MA). Best practice: choose window size based on domain knowledge (e.g., 30 days for monthly seasonality, 252 days for yearly in stock markets). Window size is a hyperparameter; validate via cross-validation. Rolling statistics are simple but powerful for exploratory analysis and feature engineering in time series projects.

---

### Q14: How do you handle missing values in time series differently from cross-sectional data?

**A:** Missing values in time series are more problematic than cross-sectional data because imputation must respect temporal structure. Simple approaches: (1) Forward fill (ffill): use last observed value yₜ = yₜ₋₁. Works for short gaps but assumes stationarity (last value remains valid). (2) Interpolation: linear, polynomial, or cubic spline interpolation estimates missing values between neighbors. Preserves trends and is less biased than forward fill. (3) Mean/median of neighbors: average values around the gap. (4) Seasonal decomposition: estimate trend, seasonal, residual separately, then combine. For longer gaps, interpolation methods often fail; best to exclude the gap or use probabilistic imputation (forecast from pre-gap data, backfill from post-gap data, take average). Advanced: Kalman filtering (sequential estimation with uncertainty) handles missing values naturally. Pitfall: avoid interpolating across fundamentally different regimes (e.g., stock market before and after circuit breaker); exclude such data instead. Never use future data to impute past (leakage). In interviews, emphasize that missing data handling is domain-specific—financial data might require different strategies than sensor data. Always document assumptions and validate imputation's impact on downstream forecasts.

---

### Q15: What is time series cross-validation and why is it different from standard cross-validation?

**A:** Standard k-fold cross-validation randomly shuffles data, violating temporal order and allowing information leakage (future data trains the model). Time series cross-validation respects temporal structure via walk-forward validation: train on [t₁, ..., tᵢ], test on [tᵢ₊₁, ..., tⱼ], then expand training window to [t₁, ..., tⱼ], test on [tⱼ₊₁, ..., tₖ], repeating until the end. No future data trains the model; evaluation mimics real-world deployment. Variants: (1) Fixed-origin CV: training window is fixed, test window expands (less realistic if data has strong trend). (2) Expanding window (growing window): training window grows, test window shifts (more realistic, but computationally expensive). (3) Rolling window: fixed-size training and test windows slide forward (balanced, most common for short-horizon forecasts). Best practice: validate the importance of temporal ordering—if results are nearly identical with shuffled cv, perhaps temporal structure is weak and standard cross-validation suffices. Time series CV is mandatory for honest evaluation; otherwise, reported accuracy is unattainably optimistic. In competitions and papers, always use time series CV for time-dependent data; this signals methodological rigor to interviewers.

---

## Interview Cheatsheet

**Key Terms:**
- **Stationarity:** Time-invariant mean, variance, autocovariance; required for classical time series methods
- **Weak Stationarity:** Constant mean, variance, and lag-dependent autocovariance; sufficient for ARIMA
- **Strict Stationarity:** Entire joint distribution invariant over time; rarely achieved in practice
- **Trend:** Long-term drift in series level; removed via differencing or detrending
- **Seasonality:** Regular periodic pattern; detected via ACF spikes at seasonal lags
- **Additive Seasonality:** yₜ = Tₜ + Sₜ + εₜ; seasonal magnitude constant across trend levels
- **Multiplicative Seasonality:** yₜ = Tₜ × Sₜ × εₜ; seasonal magnitude scales with trend
- **Autocorrelation (ACF):** Correlation between yₜ and yₜ₋ₖ; indicates temporal dependencies
- **Partial Autocorrelation (PACF):** Direct correlation at lag k after removing intermediate effects; guides AR order
- **White Noise:** Independent, identically distributed with mean 0, constant variance; target for model residuals
- **Random Walk:** yₜ = yₜ₋₁ + εₜ; non-stationary with permanent shocks; first difference yields white noise
- **Differencing:** yₜ' = yₜ - yₜ₋₁; removes trends and achieves stationarity; order d in ARIMA(p,d,q)
- **ADF Test:** Augmented Dickey-Fuller; null=unit root (non-stationary); low p-value ⟹ stationary
- **KPSS Test:** Opposite null to ADF; null=stationary; use both for robustness
- **Lag Features:** Previous values as predictors; extracted from ACF/PACF; enable supervised learning
- **Rolling Statistics:** Sliding window mean, variance, etc.; reveal trends and volatility clusters
- **Walk-Forward Validation:** Expand training, test on future; respects temporal order; prevents leakage

**Rapid-Fire Q&A:**
- **Q: How to detect non-stationarity?** **A:** Visual inspection, ADF test (unit root), KPSS test, slow ACF decay
- **Q: Difference once or twice?** **A:** Apply ADF test iteratively; one difference if unit root detected, repeat if non-stationary
- **Q: ACF/PACF interpretation?** **A:** AR(p): PACF cuts at lag p, ACF decays; MA(q): ACF cuts at lag q, PACF decays
- **Q: Additive or multiplicative?** **A:** Additive if seasonal variance constant; multiplicative if scales with trend level
- **Q: How to choose lag count?** **A:** PACF cutoff for AR order; ACF cutoff for MA order; validate with AIC/BIC
- **Q: Smooth or interpolate missing data?** **A:** Interpolation respects trends; forward fill for short gaps; Kalman for advanced cases
- **Q: Time series vs standard CV?** **A:** Time series CV uses expanding window, no future data; standard CV causes leakage
- **Q: When is white noise achieved?** **A:** Model is adequate when residuals are white noise; check ACF plot
- **Q: Exponential smoothing vs MA?** **A:** MA has fixed lag; ES weights recent data more; ES is forecasting method
- **Q: What does d mean in ARIMA?** **A:** Differencing order; d=1 removes linear trend; d=2 removes quadratic trend

---

## Interview Tips
- **Always start with diagnostics:** Plot series, ACF/PACF, run ADF and KPSS tests before any modeling
- **Leverage visual inspection:** A well-drawn ACF/PACF plot often reveals the right ARIMA order; always mention this
- **Discuss stationarity assumption:** Many candidates gloss over this; explicitly state which tests confirm stationarity achieved
- **Master walk-forward validation:** Emphasize that you use correct CV for time series; this separates practitioners from dilettantes
- **Connect to domain:** Different industries have different seasonality (retail has daily/weekly/yearly; utilities have daily/seasonal)
- **Show caution with imputation:** Never impute across regime changes; acknowledge domain-specific missing data challenges
- **Mention modern alternatives:** ARIMA is classical; mention Prophet, TBATS, deep learning for complex patterns; but master ARIMA basics first
- **Prepare visual examples:** Sketch a series with trend and seasonality, draw its decomposition, show ACF/PACF patterns for AR and MA

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*
