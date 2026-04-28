# Multiple Choice Questions: Time Series Forecasting — Advanced Methods

📺 **Video Lecture:** https://youtu.be/ziH9_Ahgq4s


Test your understanding of advanced time series forecasting methods for AI/ML interviews.

---

**Q1. In the Box-Jenkins methodology, the correct order of steps is:**

A) Examine ACF/PACF → Check stationarity → Diagnose residuals → Fit model  
B) Diagnose residuals → Check stationarity → Fit model → Examine ACF/PACF  
C) Fit model → Check stationarity → Examine ACF/PACF → Diagnose residuals  
D) Check stationarity → Examine ACF/PACF → Fit model → Diagnose residuals

---

**Q2. In ARIMA(p,d,q), the parameter d represents:**

A) The number of moving average terms  
B) The order of differencing needed to achieve stationarity  
C) The seasonal period  
D) The number of autoregressive terms

---

**Q3. SARIMA(1,1,1)(1,1,1,12) has a seasonal period of 12. The seasonal differencing component removes:**

A) Yearly periodic patterns in monthly data  
B) Quadratic trends in the data  
C) Daily cycles in hourly data  
D) Linear trends in the data

---

**Q4. Which information criterion penalizes model complexity more heavily for large sample sizes?**

A) AIC (Akaike Information Criterion)  
B) Log-likelihood  
C) BIC (Bayesian Information Criterion)  
D) R-squared

---

**Q5. A GARCH(1,1) model is primarily used to model:**

A) Missing values in time series  
B) Time-varying conditional variance (volatility clustering)  
C) The mean of a time series  
D) Seasonal patterns in sales data

---

**Q6. Facebook Prophet handles multiple seasonalities by using:**

A) Recurrent neural networks  
B) Seasonal ARIMA components  
C) Kalman filtering  
D) Fourier series terms

---

**Q7. In a VAR(p) model for two series y and x, the equation for y includes:**

A) Lagged values of both y and x  
B) Only contemporaneous values of x  
C) Only lagged values of x  
D) Only lagged values of y

---

**Q8. The Kalman filter is a sequential algorithm that provides:**

A) Seasonal decomposition only  
B) Maximum likelihood parameter estimates  
C) Optimal state estimates with uncertainty quantification  
D) Feature importance rankings

---

**Q9. What is a key disadvantage of using LSTMs for time series forecasting compared to ARIMA?**

A) LSTMs require much larger training datasets to avoid overfitting  
B) LSTMs cannot produce multi-step forecasts  
C) LSTMs assume the data is stationary  
D) LSTMs cannot handle nonlinear patterns

---

**Q10. In recursive multi-step forecasting, the main risk is:**

A) Overfitting to the test set  
B) Inability to produce probabilistic forecasts  
C) Error accumulation as predictions are fed back as inputs  
D) Requiring too many separate models

---

**Q11. MASE (Mean Absolute Scaled Error) compares forecast accuracy against:**

A) A naive baseline forecast (yₜ₋₁)  
B) A linear regression baseline  
C) The mean of the training data  
D) A perfect forecast (zero error)

---

**Q12. Granger causality tests whether:**

A) Past values of X improve prediction of Y beyond Y's own past  
B) X truly causes Y in a causal sense  
C) X and Y share the same trend  
D) X and Y are cointegrated

---

**Q13. Two non-stationary series are cointegrated if:**

A) A linear combination of them is stationary  
B) They have the same mean  
C) Both become stationary after differencing  
D) Their correlation is exactly 1

---

**Q14. Walk-forward backtesting in time series ensures:**

A) The test set is always the first 20% of data  
B) Maximum use of training data through random shuffling  
C) No future information is used during model training or evaluation  
D) Cross-validation folds overlap for robustness

---

**Q15. The Temporal Fusion Transformer (TFT) differs from standard LSTMs by providing:**

A) Only point forecasts without uncertainty  
B) Variable selection, temporal attention, and quantile regression for uncertainty  
C) Built-in seasonal decomposition like ARIMA  
D) Faster training on small datasets

---

## Answer Key

**Q1. Answer: D**
Box-Jenkins methodology follows: (1) test for stationarity and difference if needed, (2) examine ACF/PACF to identify candidate p,q orders, (3) fit the model, (4) check residual diagnostics (white noise test).

**Q2. Answer: B**
The d parameter specifies how many times the series must be differenced to achieve stationarity. Most series require d=1 (one difference removes linear trend); d=2 is rare.

**Q3. Answer: A**
With seasonal period s=12, seasonal differencing (yₜ − yₜ₋₁₂) removes yearly patterns in monthly data. This is the D=1 component in the seasonal part of SARIMA.

**Q4. Answer: C**
BIC penalizes complexity with k·ln(n), which grows with sample size n, while AIC uses a fixed 2k penalty. BIC tends to select simpler models, especially with large datasets.

**Q5. Answer: B**
GARCH models time-varying volatility (conditional variance), capturing volatility clustering where periods of high volatility tend to persist. It is widely used in financial risk management.

**Q6. Answer: D**
Prophet uses Fourier series terms to model multiple seasonalities (daily, weekly, yearly), allowing flexible periodic patterns without specifying ARIMA-style seasonal orders.

**Q7. Answer: A**
In a VAR model, each equation includes lagged values of all series in the system. This captures cross-series dependencies and enables Granger causality testing.

**Q8. Answer: C**
The Kalman filter sequentially estimates hidden states with minimum mean squared error, providing both state estimates and uncertainty (posterior covariance) at each timestep.

**Q9. Answer: A**
LSTMs have many parameters and require large training datasets (thousands of samples) to generalize well. ARIMA can work effectively with much smaller datasets due to its parsimonious parameterization.

**Q10. Answer: C**
Recursive forecasting feeds predictions back as inputs, so errors at each step compound through subsequent steps. This accumulation worsens with longer forecast horizons.

**Q11. Answer: A**
MASE scales the MAE by the MAE of a naive forecast (ŷₜ = yₜ₋₁). MASE < 1 means the model outperforms the naive baseline; MASE = 1 means equivalent performance.

**Q12. Answer: A**
Granger causality is predictive, not true causality. It tests whether past X values add statistically significant predictive power for Y beyond what Y's own past provides.

**Q13. Answer: A**
Cointegration means a linear combination of non-stationary series is stationary, implying a long-run equilibrium relationship. The individual series can each be non-stationary (e.g., both I(1)).

**Q14. Answer: C**
Walk-forward backtesting trains on past data and tests on future data sequentially, never allowing future information to leak into training. This simulates real-world deployment conditions.

**Q15. Answer: B**
TFT combines variable selection networks (identifying important features), temporal self-attention (focusing on relevant past timesteps), and quantile regression (producing uncertainty estimates), going well beyond standard LSTM capabilities.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*
