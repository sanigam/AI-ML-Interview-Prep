# Multiple Choice Questions: Time Series Fundamentals

📺 **Video Lecture:** https://youtu.be/YqHagchJ20Y


Test your understanding of time series analysis concepts essential for AI/ML interviews.

---

**Q1. Weak (covariance) stationarity requires all of the following EXCEPT:**

A) The joint distribution is identical for all time shifts  
B) Autocovariance depends only on lag, not on time  
C) Constant mean over time  
D) Constant variance over time

---

**Q2. Which plot is used to determine the order of the MA component in an ARIMA model?**

A) Q-Q plot  
B) ACF (Autocorrelation Function)  
C) Residual histogram  
D) PACF (Partial Autocorrelation Function)

---

**Q3. The Augmented Dickey-Fuller (ADF) test has the null hypothesis that:**

A) The series is stationary  
B) The series has no trend  
C) The series has a unit root (non-stationary)  
D) The series follows a normal distribution

---

**Q4. First-order differencing (yₜ − yₜ₋₁) is primarily used to:**

A) Remove seasonality from the series  
B) Compute the autocorrelation function  
C) Remove a linear trend and achieve stationarity  
D) Normalize the variance of the series

---

**Q5. A random walk yₜ = yₜ₋₁ + εₜ is non-stationary because:**

A) Its variance grows linearly with time: Var(yₜ) = t·σ²  
B) Its mean changes over time  
C) It has a deterministic trend component  
D) Its autocorrelation is always zero

---

**Q6. In an ACF plot, a slow, gradual decay of autocorrelation values typically suggests:**

A) The series is non-stationary and may need differencing  
B) The series follows a pure MA process  
C) The series is stationary  
D) The series is white noise

---

**Q7. For an AR(p) process, the PACF plot:**

A) Shows significant spikes at seasonal lags only  
B) Cuts off sharply after lag p  
C) Decays gradually to zero  
D) Is always zero for all lags

---

**Q8. The KPSS test differs from the ADF test in that:**

A) KPSS tests for normality while ADF tests for stationarity  
B) KPSS has a null hypothesis of stationarity, opposite to ADF  
C) KPSS can only be applied to seasonal data  
D) KPSS always agrees with ADF results

---

**Q9. In multiplicative decomposition yₜ = Tₜ × Sₜ × εₜ, the seasonal component Sₜ:**

A) Scales proportionally with the trend level  
B) Is always removed by first-order differencing  
C) Represents the long-term movement of the series  
D) Has constant magnitude regardless of the trend level

---

**Q10. White noise residuals from a time series model indicate that:**

A) The model has captured all predictable structure in the series  
B) The series was already stationary before modeling  
C) The model is overfitting the data  
D) The model needs more AR terms

---

**Q11. Exponential smoothing differs from a simple moving average because:**

A) It requires the series to be stationary  
B) It assigns equal weights to all past observations  
C) It assigns exponentially decreasing weights to older observations  
D) It can only handle seasonal data

---

**Q12. Seasonal differencing (yₜ − yₜ₋ₛ) is used to:**

A) Convert a multiplicative model to additive  
B) Remove a linear trend from the series  
C) Compute the standard error of forecasts  
D) Remove periodic patterns with period s

---

**Q13. When constructing lag features for a supervised learning approach to time series, which practice causes data leakage?**

A) Using future values yₜ₊₁ as a feature to predict yₜ  
B) Using seasonal lag yₜ₋₁₂ as a feature for monthly data  
C) Using yₜ₋₁ as a feature to predict yₜ  
D) Using rolling mean of past 7 values as a feature

---

**Q14. Time series cross-validation differs from standard k-fold cross-validation because:**

A) It randomly shuffles observations before splitting  
B) It always uses a fixed test set  
C) It uses fewer folds  
D) It respects temporal ordering and never trains on future data

---

**Q15. Holt-Winters exponential smoothing extends simple exponential smoothing by adding:**

A) Both trend and seasonal components  
B) An ARIMA residual correction term  
C) Only a trend component  
D) Only a seasonal component

---

## Answer Key

**Q1. Answer: A**
Weak stationarity requires constant mean, constant variance, and autocovariance depending only on lag. Requiring identical joint distributions for all time shifts is the definition of strict stationarity, which is a stronger condition.

**Q2. Answer: B**
For a pure MA(q) process, the ACF cuts off sharply after lag q, while the PACF decays gradually. The ACF cutoff directly indicates the MA order.

**Q3. Answer: C**
The ADF test's null hypothesis is that the series has a unit root (is non-stationary). Rejecting the null (low p-value) provides evidence of stationarity.

**Q4. Answer: C**
First-order differencing removes linear trends and converts an I(1) series to a stationary I(0) series. Seasonal patterns require seasonal differencing (yₜ − yₜ₋ₛ).

**Q5. Answer: A**
A random walk has variance Var(yₜ) = t·σ² which grows with time, violating the constant variance requirement for stationarity. Its mean is constant (equal to initial value) for a pure random walk without drift.

**Q6. Answer: A**
A slow ACF decay is a hallmark of non-stationarity, indicating strong persistence in the series. This typically signals the need for differencing before model fitting.

**Q7. Answer: B**
For an AR(p) process, the PACF shows significant values at lags 1 through p and then cuts off to zero, while the ACF decays gradually. This pattern guides the selection of the AR order.

**Q8. Answer: B**
The KPSS test has a null hypothesis of stationarity (opposite to ADF's null of non-stationarity). Using both tests together provides more robust stationarity diagnostics.

**Q9. Answer: A**
In multiplicative decomposition, the seasonal effect scales with the trend level. For example, a 10% seasonal increase produces larger absolute swings when the trend is higher.

**Q10. Answer: A**
White noise residuals (no autocorrelation, constant variance, zero mean) indicate the model has extracted all systematic patterns. Remaining variation is purely random and unpredictable.

**Q11. Answer: C**
Exponential smoothing assigns exponentially decreasing weights to older observations, with recent values weighted more heavily. A simple moving average assigns equal weight to all observations within the window.

**Q12. Answer: D**
Seasonal differencing subtracts the value from s periods ago, removing repeating patterns at that frequency. For monthly data with yearly seasonality, s = 12.

**Q13. Answer: A**
Using future values (yₜ₊₁) as features to predict yₜ introduces data leakage because this information would not be available at prediction time. Only past values should be used as features.

**Q14. Answer: D**
Time series cross-validation (walk-forward validation) always trains on past data and tests on future data, respecting temporal ordering. Standard k-fold randomly shuffles data, which violates temporal structure.

**Q15. Answer: A**
Holt-Winters extends exponential smoothing by adding both a trend equation and a seasonal equation, enabling it to forecast series with both trend and seasonality.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*
