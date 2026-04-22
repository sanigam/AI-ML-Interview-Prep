# Multiple Choice Questions: Time Series Forecasting — Advanced Methods

Test your understanding of advanced time series forecasting methods for AI/ML interviews.

---

**Q1. In the Box-Jenkins methodology, the correct order of steps is:**

A) Fit model → Check stationarity → Examine ACF/PACF → Diagnose residuals
B) Check stationarity → Examine ACF/PACF → Fit model → Diagnose residuals
C) Examine ACF/PACF → Check stationarity → Diagnose residuals → Fit model
D) Diagnose residuals → Check stationarity → Fit model → Examine ACF/PACF

---

**Q2. In ARIMA(p,d,q), the parameter d represents:**

A) The number of autoregressive terms
B) The order of differencing needed to achieve stationarity
C) The number of moving average terms
D) The seasonal period

---

**Q3. SARIMA(1,1,1)(1,1,1,12) has a seasonal period of 12. The seasonal differencing component removes:**

A) Linear trends in the data
B) Yearly periodic patterns in monthly data
C) Daily cycles in hourly data
D) Quadratic trends in the data

---

**Q4. Which information criterion penalizes model complexity more heavily for large sample sizes?**

A) AIC (Akaike Information Criterion)
B) BIC (Bayesian Information Criterion)
C) Log-likelihood
D) R-squared

---

**Q5. A GARCH(1,1) model is primarily used to model:**

A) The mean of a time series
B) Time-varying conditional variance (volatility clustering)
C) Seasonal patterns in sales data
D) Missing values in time series

---

**Q6. Facebook Prophet handles multiple seasonalities by using:**

A) Seasonal ARIMA components
B) Fourier series terms
C) Kalman filtering
D) Recurrent neural networks

---

**Q7. In a VAR(p) model for two series y and x, the equation for y includes:**

A) Only lagged values of y
B) Only lagged values of x
C) Lagged values of both y and x
D) Only contemporaneous values of x

---

**Q8. The Kalman filter is a sequential algorithm that provides:**

A) Maximum likelihood parameter estimates
B) Optimal state estimates with uncertainty quantification
C) Feature importance rankings
D) Seasonal decomposition only

---

**Q9. What is a key disadvantage of using LSTMs for time series forecasting compared to ARIMA?**

A) LSTMs cannot handle nonlinear patterns
B) LSTMs require much larger training datasets to avoid overfitting
C) LSTMs cannot produce multi-step forecasts
D) LSTMs assume the data is stationary

---

**Q10. In recursive multi-step forecasting, the main risk is:**

A) Requiring too many separate models
B) Error accumulation as predictions are fed back as inputs
C) Inability to produce probabilistic forecasts
D) Overfitting to the test set

---

**Q11. MASE (Mean Absolute Scaled Error) compares forecast accuracy against:**

A) A perfect forecast (zero error)
B) A naive baseline forecast (yₜ₋₁)
C) The mean of the training data
D) A linear regression baseline

---

**Q12. Granger causality tests whether:**

A) X truly causes Y in a causal sense
B) Past values of X improve prediction of Y beyond Y's own past
C) X and Y are cointegrated
D) X and Y share the same trend

---

**Q13. Two non-stationary series are cointegrated if:**

A) Both become stationary after differencing
B) A linear combination of them is stationary
C) They have the same mean
D) Their correlation is exactly 1

---

**Q14. Walk-forward backtesting in time series ensures:**

A) Maximum use of training data through random shuffling
B) No future information is used during model training or evaluation
C) The test set is always the first 20% of data
D) Cross-validation folds overlap for robustness

---

**Q15. The Temporal Fusion Transformer (TFT) differs from standard LSTMs by providing:**

A) Only point forecasts without uncertainty
B) Variable selection, temporal attention, and quantile regression for uncertainty
C) Faster training on small datasets
D) Built-in seasonal decomposition like ARIMA

---

## Answer Key

**Q1. Answer: B**
Box-Jenkins methodology follows: (1) test for stationarity and difference if needed, (2) examine ACF/PACF to identify candidate p,q orders, (3) fit the model, (4) check residual diagnostics (white noise test).

**Q2. Answer: B**
The d parameter specifies how many times the series must be differenced to achieve stationarity. Most series require d=1 (one difference removes linear trend); d=2 is rare.

**Q3. Answer: B**
With seasonal period s=12, seasonal differencing (yₜ − yₜ₋₁₂) removes yearly patterns in monthly data. This is the D=1 component in the seasonal part of SARIMA.

**Q4. Answer: B**
BIC penalizes complexity with k·ln(n), which grows with sample size n, while AIC uses a fixed 2k penalty. BIC tends to select simpler models, especially with large datasets.

**Q5. Answer: B**
GARCH models time-varying volatility (conditional variance), capturing volatility clustering where periods of high volatility tend to persist. It is widely used in financial risk management.

**Q6. Answer: B**
Prophet uses Fourier series terms to model multiple seasonalities (daily, weekly, yearly), allowing flexible periodic patterns without specifying ARIMA-style seasonal orders.

**Q7. Answer: C**
In a VAR model, each equation includes lagged values of all series in the system. This captures cross-series dependencies and enables Granger causality testing.

**Q8. Answer: B**
The Kalman filter sequentially estimates hidden states with minimum mean squared error, providing both state estimates and uncertainty (posterior covariance) at each timestep.

**Q9. Answer: B**
LSTMs have many parameters and require large training datasets (thousands of samples) to generalize well. ARIMA can work effectively with much smaller datasets due to its parsimonious parameterization.

**Q10. Answer: B**
Recursive forecasting feeds predictions back as inputs, so errors at each step compound through subsequent steps. This accumulation worsens with longer forecast horizons.

**Q11. Answer: B**
MASE scales the MAE by the MAE of a naive forecast (ŷₜ = yₜ₋₁). MASE < 1 means the model outperforms the naive baseline; MASE = 1 means equivalent performance.

**Q12. Answer: B**
Granger causality is predictive, not true causality. It tests whether past X values add statistically significant predictive power for Y beyond what Y's own past provides.

**Q13. Answer: B**
Cointegration means a linear combination of non-stationary series is stationary, implying a long-run equilibrium relationship. The individual series can each be non-stationary (e.g., both I(1)).

**Q14. Answer: B**
Walk-forward backtesting trains on past data and tests on future data sequentially, never allowing future information to leak into training. This simulates real-world deployment conditions.

**Q15. Answer: B**
TFT combines variable selection networks (identifying important features), temporal self-attention (focusing on relevant past timesteps), and quantile regression (producing uncertainty estimates), going well beyond standard LSTM capabilities.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*
