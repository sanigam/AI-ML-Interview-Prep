# Multiple Choice Questions: Time Series Fundamentals

📺 **Video Lecture:** https://youtu.be/YqHagchJ20Y


Test your understanding of time series analysis concepts essential for AI/ML interviews.

---

**Q1. Weak (covariance) stationarity requires all of the following EXCEPT:**

A) Constant mean over time  
B) Constant variance over time  
C) Autocovariance depends only on lag, not on time  
D) The joint distribution is identical for all time shifts

---

**Q2. Which plot is used to determine the order of the MA component in an ARIMA model?**

A) PACF (Partial Autocorrelation Function)  
B) ACF (Autocorrelation Function)  
C) Q-Q plot  
D) Residual histogram

---

**Q3. The Augmented Dickey-Fuller (ADF) test has the null hypothesis that:**

A) The series is stationary  
B) The series has a unit root (non-stationary)  
C) The series has no trend  
D) The series follows a normal distribution

---

**Q4. First-order differencing (yₜ − yₜ₋₁) is primarily used to:**

A) Remove seasonality from the series  
B) Remove a linear trend and achieve stationarity  
C) Normalize the variance of the series  
D) Compute the autocorrelation function

---

**Q5. A random walk yₜ = yₜ₋₁ + εₜ is non-stationary because:**

A) Its mean changes over time  
B) Its variance grows linearly with time: Var(yₜ) = t·σ²  
C) Its autocorrelation is always zero  
D) It has a deterministic trend component

---

**Q6. In an ACF plot, a slow, gradual decay of autocorrelation values typically suggests:**

A) The series is white noise  
B) The series is stationary  
C) The series is non-stationary and may need differencing  
D) The series follows a pure MA process

---

**Q7. For an AR(p) process, the PACF plot:**

A) Decays gradually to zero  
B) Cuts off sharply after lag p  
C) Shows significant spikes at seasonal lags only  
D) Is always zero for all lags

---

**Q8. The KPSS test differs from the ADF test in that:**

A) KPSS tests for normality while ADF tests for stationarity  
B) KPSS has a null hypothesis of stationarity, opposite to ADF  
C) KPSS can only be applied to seasonal data  
D) KPSS always agrees with ADF results

---

**Q9. In multiplicative decomposition yₜ = Tₜ × Sₜ × εₜ, the seasonal component Sₜ:**

A) Has constant magnitude regardless of the trend level  
B) Scales proportionally with the trend level  
C) Is always removed by first-order differencing  
D) Represents the long-term movement of the series

---

**Q10. White noise residuals from a time series model indicate that:**

A) The model is overfitting the data  
B) The model has captured all predictable structure in the series  
C) The model needs more AR terms  
D) The series was already stationary before modeling

---

**Q11. Exponential smoothing differs from a simple moving average because:**

A) It assigns equal weights to all past observations  
B) It assigns exponentially decreasing weights to older observations  
C) It can only handle seasonal data  
D) It requires the series to be stationary

---

**Q12. Seasonal differencing (yₜ − yₜ₋ₛ) is used to:**

A) Remove a linear trend from the series  
B) Remove periodic patterns with period s  
C) Compute the standard error of forecasts  
D) Convert a multiplicative model to additive

---

**Q13. When constructing lag features for a supervised learning approach to time series, which practice causes data leakage?**

A) Using yₜ₋₁ as a feature to predict yₜ  
B) Using rolling mean of past 7 values as a feature  
C) Using future values yₜ₊₁ as a feature to predict yₜ  
D) Using seasonal lag yₜ₋₁₂ as a feature for monthly data

---

**Q14. Time series cross-validation differs from standard k-fold cross-validation because:**

A) It uses fewer folds  
B) It respects temporal ordering and never trains on future data  
C) It always uses a fixed test set  
D) It randomly shuffles observations before splitting

---

**Q15. Holt-Winters exponential smoothing extends simple exponential smoothing by adding:**

A) Only a trend component  
B) Only a seasonal component  
C) Both trend and seasonal components  
D) An ARIMA residual correction term

---

## Answer Key

**Q1. Answer: D**
Weak stationarity requires constant mean, constant variance, and autocovariance depending only on lag. Requiring identical joint distributions for all time shifts is the definition of strict stationarity, which is a stronger condition.

**Q2. Answer: B**
For a pure MA(q) process, the ACF cuts off sharply after lag q, while the PACF decays gradually. The ACF cutoff directly indicates the MA order.

**Q3. Answer: B**
The ADF test's null hypothesis is that the series has a unit root (is non-stationary). Rejecting the null (low p-value) provides evidence of stationarity.

**Q4. Answer: B**
First-order differencing removes linear trends and converts an I(1) series to a stationary I(0) series. Seasonal patterns require seasonal differencing (yₜ − yₜ₋ₛ).

**Q5. Answer: B**
A random walk has variance Var(yₜ) = t·σ² which grows with time, violating the constant variance requirement for stationarity. Its mean is constant (equal to initial value) for a pure random walk without drift.

**Q6. Answer: C**
A slow ACF decay is a hallmark of non-stationarity, indicating strong persistence in the series. This typically signals the need for differencing before model fitting.

**Q7. Answer: B**
For an AR(p) process, the PACF shows significant values at lags 1 through p and then cuts off to zero, while the ACF decays gradually. This pattern guides the selection of the AR order.

**Q8. Answer: B**
The KPSS test has a null hypothesis of stationarity (opposite to ADF's null of non-stationarity). Using both tests together provides more robust stationarity diagnostics.

**Q9. Answer: B**
In multiplicative decomposition, the seasonal effect scales with the trend level. For example, a 10% seasonal increase produces larger absolute swings when the trend is higher.

**Q10. Answer: B**
White noise residuals (no autocorrelation, constant variance, zero mean) indicate the model has extracted all systematic patterns. Remaining variation is purely random and unpredictable.

**Q11. Answer: B**
Exponential smoothing assigns exponentially decreasing weights to older observations, with recent values weighted more heavily. A simple moving average assigns equal weight to all observations within the window.

**Q12. Answer: B**
Seasonal differencing subtracts the value from s periods ago, removing repeating patterns at that frequency. For monthly data with yearly seasonality, s = 12.

**Q13. Answer: C**
Using future values (yₜ₊₁) as features to predict yₜ introduces data leakage because this information would not be available at prediction time. Only past values should be used as features.

**Q14. Answer: B**
Time series cross-validation (walk-forward validation) always trains on past data and tests on future data, respecting temporal ordering. Standard k-fold randomly shuffles data, which violates temporal structure.

**Q15. Answer: C**
Holt-Winters extends exponential smoothing by adding both a trend equation and a seasonal equation, enabling it to forecast series with both trend and seasonality.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*
