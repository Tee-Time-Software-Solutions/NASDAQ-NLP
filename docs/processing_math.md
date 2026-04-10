1. Simple (daily) returns
For each asset, daily return from Adjusted Close (splits/dividends handled in the price series): [ R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1 ]
Same idea for the stock and for the NASDAQ index.
2. Event time (relative trading time)
Each earnings call is anchored to a trading day (t = 0) (event_trading_day_final).
Other days are indexed as integers (t = -120,\ldots,-1,0,1,\ldots) in trading days, not calendar days.
3. Linear regression / ordinary least squares (OLS) — market model
In the estimation window (t \in [-120, -20]), for each event you estimate: [ R_{i,t} = \alpha_i + \beta_i R_{m,t} + \varepsilon_{i,t} ] where (R_{i,t}) is the stock return and (R_{m,t}) is the market (NASDAQ) return.
(\beta_i) is the stock’s sensitivity to the market; (\alpha_i) is the intercept; (\varepsilon_{i,t}) is the regression residual.
4. Expected return under the market model
Using the estimated (\hat\alpha_i, \hat\beta_i), the model-implied expected return on day (t) is: [ E[R_{i,t} \mid R_{m,t}] = \hat\alpha_i + \hat\beta_i R_{m,t} ] (same structure as in your abnormal-return notebook).
5. Abnormal return (AR)
The abnormal return is the difference between realized return and the market-model prediction: [ AR_{i,t} = R_{i,t} - \big(\hat\alpha_i + \hat\beta_i R_{m,t}\big) ]
Conceptually this is “return not explained by the market model” in that window.
6. Cumulative abnormal return (CAR)
CAR over a set of event-time days (\mathcal{W}) is the sum of abnormal returns on those days: [ CAR_i(\mathcal{W}) = \sum_{t \in \mathcal{W}} AR_{i,t} ]
Your project uses e.g.:
CAR[0,1] = (AR_{i,0} + AR_{i,1})
CAR[0,3] = (AR_{i,0} + AR_{i,1} + AR_{i,2} + AR_{i,3})
7. Volatility as standard deviation of returns
Over a window of trading days, volatility is typically the sample standard deviation of daily returns in that window (often treating those days as the sample for that event).
8. Volatility change (difference of volatilities)
You compare pre- vs post-event volatility over two windows in event time, e.g.:
Pre: (t \in [-10,-1])
Post: (t \in [+1,+10])
Change is a difference: [ \Delta \text{Vol}_i = \text{Vol}_i^{\text{post}} - \text{Vol}_i^{\text{pre}} ] (as in your volatility notebook’s definition).
9. Business-day / calendar logic (supporting math, not regression)
Next business day (weekday skip) when assigning the event trading day after the close — this is discrete calendar/business-day arithmetic, not continuous-time finance.