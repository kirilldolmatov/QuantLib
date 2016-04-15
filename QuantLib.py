
# coding: utf-8

# # Install
# 1. Download "WinPython" from https://sourceforge.net/projects/winpython/files/WinPython_3.4/3.4.4.1/WinPython-64bit-3.4.4.1.exe/download 
# 2. Download "quantlib" from http://www.lfd.uci.edu/~gohlke/pythonlibs/
# 3. Open "quantlib.whl" package with "WinPython Control Panel"    

# ### Some features
# * TARGET - Trans-European Automated Real-time Gross settlement Express Transfer
# * For all constants (spot, vol and etc.) recommend to use "SimpleQuote" then it's reference

# In[1]:

# Import QuantLib. 
import QuantLib as ql


# In[2]:

# Import other lib's
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt


# In[3]:

# Some magic for plot


# In[4]:

get_ipython().magic('matplotlib inline')


# ---

# # DateTime

# ### Set the evaluation time
# It's required for futher actions

# In[5]:

evaluationDate = ql.Date(15, 1, 2015)
ql.Settings.instance().evaluationDate = evaluationDate


# ### DateTime Properties

# In[6]:

date = ql.Date(30, 3, 2016)
print (date)


# In[7]:

print("Year = %s, Month = %s, DayOfMonth = %s, Tuesday = %s" % 
      (date.year(), date.month(), date.dayOfMonth(), date.weekday() == ql.Tuesday))


# ### DateTime Arithmetic

# In[8]:

print(date + 1)
print(date - 1)
print(date + ql.Period(1, ql.Days))
print(date + ql.Period(1, ql.Weeks))
print(date + ql.Period(1, ql.Years))
print(date + 1 > date - 1)


# ### DateTime Schedule (ex. Coupon Payments)

# Here we have generated a Schedule object that will contain dates between date1 and date2 with the tenor specifying the Period to be every Month. The calendar object is used for determining holidays. The two arguments following the calendar in the Schedule constructor are the BussinessDayConvention. Here we chose the convention to be the day following holidays. That is why we see that holidays are excluded in the list of dates.

# In[9]:

date1 = ql.Date(1, 1, 2016)
date2 = ql.Date(1, 1, 2017)
tenor = ql.Period(ql.Monthly)
calendar = ql.Russia()
schedule = ql.Schedule(date1, date2, tenor, calendar, ql.Following, ql.Following, ql.DateGeneration.Forward, False)
list(schedule)


# ---

# # Interest Rate

# The InterestRate class can be used to store the interest rate with the compounding type, day count and the frequency of compounding. Below we show how to create an interest rate of 5.0% compounded annually, using Actual/Actual day count convention. Here you can see more about [day count convention.](https://en.wikipedia.org/wiki/Day_count_convention)

# In[10]:

annualRate = 0.05
dayCount = ql.ActualActual()
compoundType = ql.Compounded
frequency = ql.Annual
interestRate = ql.InterestRate(annualRate, dayCount, compoundType, frequency)


# Lets say if you invest a dollar at the interest rate described by interestRate, the compoundFactor method gives you how much your investment will be worth after t years. Below we show that the value returned by compoundFactor for 2 years agrees with the expected compounding formula.

# In[11]:

print ((1.0 + annualRate)*(1.0 + annualRate)) # cum-rate for 2 years 
print(interestRate.compoundFactor(2.0)) 


# The discountFactor method returns the reciprocal of the compoundFactor method. The discount factor is useful while calculating the present value of future cashflows.

# In[12]:

print(1.0 / interestRate.compoundFactor(2.0))
print(interestRate.discountFactor(2.0))


# Compare interest rates that have different compounding periods and/or payment frequencies.

# In[13]:

newCompoundType = ql.Continuous
newRate = interestRate.equivalentRate(newCompoundType, frequency, 1)
print ("Rate with annual compounding is: %s, Equivalent one year annual continuously compounded rate is: %s" %
       (interestRate.rate(), newRate.rate()))

newFrequency = ql.Semiannual
newRate = interestRate.equivalentRate(compoundType, newFrequency, 1)
print ("Rate with annual compounding is: %s, Equivalent one year semi-annually compounded rate is: %s" %
       (interestRate.rate(), newRate.rate()))


# ---

# # Modeling Fixed Rate Bonds

# Let's consider a hypothetical bond with a par value of 100, that pays 6% coupon semi-annually issued on January 15th, 2015 and set to mature on January 15th, 2016. The bond will pay a coupon on July 15th, 2015 and January 15th, 2016. The par amount of 100 will also be paid on the January 15th, 2016.
# 
# To make things simpler, lets assume that we know the spot rates of the treasury as of January 15th, 2015. The annualized spot rates are 0.5% for 6 months and 0.7% for 1 year point. Lets calculate the fair value of this bond.

# $$ P=(\frac{C}{1+i} + \frac{C}{(1+i)^2} + \cdots + \frac{C}{(1+i)^N}) + \frac{M}{(1+i)^N} $$    

# In[6]:

P = (3 / pow(1 + 0.005, 0.5)) + (3 / (1 + 0.007) + 100 / (1 + 0.007))
print (P)


# Lets calculate the same thing using QuantLib.

# In[7]:

spotDates = [ql.Date(15, 1, 2015), ql.Date(15, 7, 2015), ql.Date(15, 1, 2016)]
spotRates = [0.0, 0.005, 0.007]
dayCount = ql.Thirty360()
calendar = ql.Russia()
interpolation = ql.Linear()
compounding = ql.Compounded
compoundingFrequency = ql.Annual
spotCurve = ql.ZeroCurve(spotDates, spotRates, dayCount, calendar, interpolation, compounding, compoundingFrequency)
spotCurveHandle = ql.YieldTermStructureHandle(spotCurve)


# So far we have created the term structure and the variables are rather self explanatory. Now lets construct the schedule for bond.

# In[8]:

issueDate = ql.Date(15, 1, 2015)
maturityDate = ql.Date(15, 1, 2016)
tenor = ql.Period(ql.Semiannual)
bussinessConvention = ql.Unadjusted
dateGeneration = ql.DateGeneration.Backward
monthEnd = False
schedule = ql.Schedule(issueDate, maturityDate, tenor, calendar, 
                    bussinessConvention, bussinessConvention, dateGeneration, monthEnd)
list(schedule)


# Now lets build the coupon

# In[9]:

couponRate = 0.06
coupons = [couponRate]


# Now lets construct the FixedRateBond

# In[126]:

settlementDays = 0
faceValue = 100
fixedRateBond = ql.FixedRateBond(settlementDays, faceValue, schedule, coupons, dayCount)


# Create a bond engine with the term structure as input and set the bond to use this bond engine

# In[28]:

bondEngine = ql.DiscountingBondEngine(spotCurveHandle)
fixedRateBond.setPricingEngine(bondEngine)


# Finally the price

# In[29]:

fixedRateBond.NPV()


# In[31]:

print("Manually Calc: %s \nQuantlib Calc: %s" % (P, fixedRateBond.NPV()))


# Cash Flows

# In[33]:

for i, cf in enumerate(fixedRateBond.cashflows()):
    print ("%2d  %-20s  %10.2f" % (i+1, cf.date(), cf.amount()))


# ---

# # Interest Rate Term Structure

# Term structure is pivotal to pricing securities. One would need a YieldTermStructure object created in QuantLib to use with pricing engines. In an earlier example on modeling bonds using QuantLib we discussed how to use spot rates directly with bond pricing engine. Here we will show how to bootstrap yield curve using QuantLib.

# The deposit rates and fixed rate bond rates are provided below.

# In[48]:

depo_maturities = [ql.Period(6, ql.Months), ql.Period(12, ql.Months)]
depo_rates = [5.25, 5.5]

bond_maturities = [ql.Period(6*i, ql.Months) for i in range(3,21)]
bond_rates = [5.75, 6.0, 6.25, 6.5, 6.75, 6.80, 7.00, 7.1, 7.15, 7.2, 7.3, 7.35, 7.4, 7.5, 7.6, 7.6, 7.7, 7.8]


# Lets define some of the constants required for the rest of the objects needed below.

# In[49]:

calendar = ql.Russia()
bussiness_convention = ql.Unadjusted
day_count = ql.Thirty360()
end_of_month = True
settlement_days = 0
face_amount = 100
coupon_frequency = ql.Period(ql.Semiannual)
settlement_days = 0


# The basic idea of bootstrapping using QuantLib is to use the deposit rates and bond rates to create individual helpers. Then use the combination of the two helpers to construct the yield curve.

# In[50]:

depo_helpers = [ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(r/100.0)),
                                     m,
                                     settlement_days,
                                     calendar,
                                     bussiness_convention,
                                     end_of_month,
                                     day_count )
                for r, m in zip(depo_rates, depo_maturities)]


# The rest of the points are coupon bonds. We assume that the YTM given for the bonds are all par rates. So we have bonds with coupon rate same as the YTM.

# In[51]:

bond_helpers = []
for r, m in zip(bond_rates, bond_maturities):
    termination_date = evaluationDate + m
    schedule = ql.Schedule(evaluationDate,
                   termination_date,
                   coupon_frequency,
                   calendar,
                   bussiness_convention,
                   bussiness_convention,
                   ql.DateGeneration.Backward,
                   end_of_month)

    helper = ql.FixedRateBondHelper(ql.QuoteHandle(ql.SimpleQuote(face_amount)),
                                        settlement_days,
                                        face_amount,
                                        schedule,
                                        [r/100.0],
                                        day_count,
                                        bussiness_convention)
    bond_helpers.append(helper)


# The yield curve is constructed by putting the two helpers together.

# In[53]:

rate_helpers = depo_helpers + bond_helpers
yieldcurve = ql.PiecewiseLogCubicDiscount(evaluationDate, rate_helpers, day_count)


# The spot rates is obtined from yieldcurve object using the forwardRate method.

# In[45]:

spots = []
for d in yieldcurve.dates():    
    compounding = ql.Compounded
    freq = ql.Semiannual
    forward_rate = yieldcurve.forwardRate(evaluationDate, d, day_count, compounding, freq)
    eq_rate = forward_rate.equivalentRate(day_count,
                                       compounding,
                                       freq,
                                       evaluationDate,
                                       d).rate()
    spots.append(100*eq_rate)
        
yield_by_date = list(zip(yieldcurve.dates(), spots))    
yield_by_date


# The spot rates is obtined from yieldcurve object using the zeroRate method.

# In[47]:

spots = []
for d in yieldcurve.dates():
    yrs = day_count.yearFraction(evaluationDate, d)
    compounding = ql.Compounded
    freq = ql.Semiannual
    zero_rate = yieldcurve.zeroRate(yrs, compounding, freq)
    eq_rate = zero_rate.equivalentRate(day_count,
                                       compounding,
                                       freq,
                                       evaluationDate,
                                       d).rate()
    spots.append(100*eq_rate)
    
yield_by_date = list(zip(yieldcurve.dates(),spots))    
yield_by_date


# ---

# # Hull White Term Structure Simulations

# The Hull-White Short Rate Model is defined as:
# 
# $$ dr_t = (\theta(t) - a r_t)dt + \sigma dW_t $$
# 
# where $a$ and $ \sigma $ are constants, and $\theta(t)$ is
# chosen in order to fit the input term structure of interest rates.
# Here we use QuantLib to show how to simulate the Hull-White model
# and investigate some of the properties.

# The constants that we use for this example is all defined as shown below. Variables $sigma$ and $a$ are the constants that define the Hull-White model. In the simulation, we discretize the time span of length 30 years into 360 intervals (one per month) as defined by the timestep variable. For simplicity we will use a constant forward rate term structure as an input. It is straight forward to swap with another term structure here.

# In[61]:

sigma = 0.1
a = 0.1
timestep = 360
length = 30 # in years
forward_rate = 0.05
day_count = ql.Thirty360()


# In[62]:

spot_curve = ql.FlatForward(evaluationDate, ql.QuoteHandle(ql.SimpleQuote(forward_rate)), day_count)
spot_curve_handle = ql.YieldTermStructureHandle(spot_curve)


# In[63]:

hw_process = ql.HullWhiteProcess(spot_curve_handle, a, sigma)
rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
seq = ql.GaussianPathGenerator(hw_process, length, timestep, rng, False)


# The Hull-White process is constructed by passing the term-structure, $a$ and $sigma$. To create the path generator, one has to provide a random sequence generator along with other simulation inputs such as timestep and length.
# A function to generate paths can be written as shown below:

# In[64]:

def generate_paths(num_paths, timestep):
    arr = np.zeros((num_paths, timestep+1))
    for i in range(num_paths):
        sample_path = seq.next()
        path = sample_path.value()
        time = [path.time(j) for j in range(len(path))]
        value = [path[j] for j in range(len(path))]
        arr[i, :] = np.array(value)
    return np.array(time), arr


# The simulation of the short rates look as shown below:

# In[65]:

num_paths = 100
time, paths = generate_paths(num_paths, timestep)
for i in range(num_paths):
    plt.plot(time, paths[i, :], lw=0.8, alpha=0.6)
plt.title("Hull-White Short Rate Simulation")
plt.show()


# The short rate $r(t)$ is given a distribution with the properties:
# 
# $$ E\{r(t) | F_s\} = r(s)e^{-a(t-s)}  + \alpha(t) - \alpha(s)e^{-a(t-s)} $$
#    $$ Var\{ r(t) | F_s \} = \frac{\sigma^2}{2a} [1 - e^{-2a(t-s)}] $$
#    where 
#    $$ \alpha(t) = f^M(0, t) + \frac{\sigma^2} {2a^2}(1-e^{-at})^2$$
#    
# as shown in Brigo & Mercurio's book on Interest Rate Models.

# In[68]:

num_paths = 1000
time, paths = generate_paths(num_paths, timestep)


# The mean and variance compared between the simulation (red dotted line) and theory (blue line).

# In[69]:

vol = [np.var(paths[:, i]) for i in range(timestep+1)]
plt.plot(time, vol, "r-.", lw=3, alpha=0.6)
plt.plot(time,sigma*sigma/(2*a)*(1.0-np.exp(-2.0*a*np.array(time))), "b-", lw=2, alpha=0.5)
plt.title("Variance of Short Rates")


# In[70]:

def alpha(forward, sigma, a, t):
    return forward + 0.5* np.power(sigma/a*(1.0 - np.exp(-a*t)), 2)

avg = [np.mean(paths[:, i]) for i in range(timestep+1)]
plt.plot(time, avg, "r-.", lw=3, alpha=0.6)
plt.plot(time,alpha(forward_rate, sigma, a, time), "b-", lw=2, alpha=0.6)
plt.title("Mean of Short Rates")


# ---

# # European Option Pricing using Black-Scholes-Merton model

# Let us consider a European call option for AAPL with a strike price of \$130 maturing on 15th Jan, 2016. Let the spot price be \$127.62. The volatility of the underlying stock is know to be 20%, and has a dividend yield of 1.63%. Lets value this option as of evaluationDate.

# In[71]:

maturity_date = ql.Date(15, 1, 2016)
spot_price = 127.62
strike_price = 130
volatility = 0.20 # the historical vols for a year
dividend_rate =  0.0163
option_type = ql.Option.Call

risk_free_rate = 0.001
day_count = ql.Actual365Fixed()
calendar = ql.TARGET() # Trans-European Automated Real-time Gross settlement Express Transfer


# Construct the European option

# In[72]:

payoff = ql.PlainVanillaPayoff(option_type, strike_price)
exercise = ql.EuropeanExercise(maturity_date)
european_option = ql.VanillaOption(payoff, exercise)


# Construct The Black-Scholes-Merton process

# In[73]:

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(evaluationDate, risk_free_rate, day_count))
dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(evaluationDate, dividend_rate, day_count))
flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(evaluationDate, calendar, volatility, day_count))
bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts)


# Lets compute the theoretical price using the AnalyticEuropeanEngine

# In[74]:

engine = ql.AnalyticEuropeanEngine(bsm_process)
european_option.setPricingEngine(engine)
bs_price = european_option.NPV()
print ("The theoretical price is:", bs_price)


# ---

# # European Option Pricing using Binominal-tree approach

# In[75]:

def binomial_price(bsm_process, steps):
    binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
    european_option.setPricingEngine(binomial_engine)
    return european_option.NPV()

steps = range(2, 100, 1)
prices = [binomial_price(bsm_process, step) for step in steps]


# In the plot below, we show the convergence of binomial-tree approach by comparing its price with the BSM price.

# In[76]:

plt.plot(steps, prices, label="Binomial Tree Price", lw=2, alpha=0.6)
plt.plot([0,100],[bs_price, bs_price], "r--", label="BSM Price", lw=2, alpha=0.6)
plt.xlabel("Steps")
plt.ylabel("Price")
plt.title("Binomial Tree Price For Varying Steps")
plt.legend()


# ---

# # European Option Pricing using Black model

# Let us consider a European call option for AAPL with a strike price of \$130 maturing on 15th Jan, 2016. Let the spot price be \$127.62. The volatility of the underlying stock is know to be 20%. Lets value this option as of evaluationDate.

# In[77]:

maturity_date = ql.Date(15, 1, 2016)
spot = 127.62
strike = 130
volatility = 0.20 # the historical vols for a year
option_type = ql.Option.Call

risk_free_rate = 0.001
day_count = ql.Actual365Fixed()
calendar = ql.TARGET() # Trans-European Automated Real-time Gross settlement Express Transfer


# Construct the European option

# In[78]:

payoff = ql.PlainVanillaPayoff(option_type, strike)
exercise = ql.EuropeanExercise(maturity_date)
option = ql.VanillaOption(payoff, exercise)


# Construct The Black process

# In[79]:

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(evaluationDate, risk_free_rate, day_count))
volatitlity = ql.BlackConstantVol(evaluationDate, calendar, volatility, day_count)
b_process = ql.BlackProcess(spot_handle, risk_free_ts, ql.BlackVolTermStructureHandle(volatitlity))


# Lets compute the theoretical price using the AnalyticEuropeanEngine

# In[80]:

engine = ql.AnalyticEuropeanEngine(b_process)


# In[81]:

option.setPricingEngine(engine)


# In[82]:

print (option.NPV())
print (option.delta())
print (option.gamma())
print (option.vega())
print (option.rho())
print (option.theta())


# ---

# # European Option Pricing using Garman-Kohlhagen model

# Let us consider a European call option for AAPL with a strike price of \$130 maturing on 15th Jan, 2016. Let the spot price be \$127.62. The volatility of the underlying stock is know to be 20%. Lets value this option as of evaluationDate.

# In[83]:

maturity_date = ql.Date(15, 1, 2016)
spot = 127.62
strike = 130
volatility = 0.20 # the historical vols for a year
option_type = ql.Option.Call

risk_free_rate_foreign = 0.011
risk_free_rate_domestic = 0.001
day_count = ql.Actual365Fixed()
calendar = ql.TARGET() # Trans-European Automated Real-time Gross settlement Express Transfer


# Construct the European option

# In[84]:

payoff = ql.PlainVanillaPayoff(option_type, strike)
exercise = ql.EuropeanExercise(maturity_date)
option = ql.VanillaOption(payoff, exercise)


# Construct the Garman-Kohlhagen process

# In[85]:

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

foreign_risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(evaluationDate, risk_free_rate_foreign, day_count))

domestic_risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(evaluationDate, risk_free_rate_domestic, day_count))

volatitlity = ql.BlackConstantVol(evaluationDate, calendar, volatility, day_count)

gk_process = ql.GarmanKohlagenProcess(spot_handle, foreign_risk_free_ts, domestic_risk_free_ts, ql.BlackVolTermStructureHandle(volatitlity))


# Lets compute the theoretical price using the AnalyticEuropeanEngine

# In[86]:

engine = ql.AnalyticEuropeanEngine(gk_process)


# In[87]:

option.setPricingEngine(engine)


# In[88]:

print (option.NPV())
print (option.delta())
print (option.gamma())
print (option.vega())
print (option.rho())
print (option.theta())


# ---

# # European Option Pricing using Heston model

# Heston model can be used to value options by modeling the underlying asset such as the stock of a company. The one major feature of the Heston model is that it inocrporates a stochastic volatility term.
# 
# \begin{eqnarray}
# dS_t &=& \mu S_tdt + \sqrt{V_t} S_t dW_t^1 \\
# dV_t &=& \kappa(\theta-V_t) + \sigma \sqrt{V_t} dW_t^2
# \end{eqnarray}
# 
# * $S_t$  is the asset's value at time tt
# * $\mu$ is the expected growth rate of the log normal stock value
# * $V_t$ is the variance of the asset $S_t$
# * $W_t^1$ is the stochastic process governing the $S_t$ process
# * $\theta$ is the value of mean reversion for the variance $V_t$
# * $\kappa$ is the strengh of mean reversion
# * $\sigma$ is the volatility of volatility
# * $W_t^2$ is the stochastic process governing the $V_t$ process
# * The correlation between $W_t^1$ and $W_t^2$ is $ρ$
# 
# In contrast, the Black-Scholes-Merton process assumes that the volatility is constant.

# Let us consider a European call option for AAPL with a strike price of \$130 maturing on 15th Jan, 2016. Let the spot price be \$127.62. The volatility of the underlying stock is know to be 20%, and has a dividend yield of 1.63%. We assume a short term risk free rate of 0.1%. Lets value this option as of 8th May, 2015.

# In[93]:

maturity_date = ql.Date(15, 1, 2016)
spot = ql.SimpleQuote(127.62)
strike = 130
volatility = 0.20 # the historical vols for a year
dividend_rate = 0.0163
option_type = ql.Option.Call

risk_free_rate = 0.001
day_count = ql.Actual365Fixed()
calendar = ql.TARGET() # Trans-European Automated Real-time Gross settlement Express Transfer


# Construct the European option

# In[94]:

payoff = ql.PlainVanillaPayoff(option_type, strike)
exercise = ql.EuropeanExercise(maturity_date)
option = ql.VanillaOption(payoff, exercise)


# In order to price the option using the Heston model, we first create the Heston process. In order to create the Heston process, we use the parameter values: 
# * the spot variance v0 = volatility*volatility = 0.04
# * mean reversion strength kappa = 0.1
# * the mean reversion variance theta=v0
# * volatility of volatility sigma = 0.1 
# * the correlation between the asset price and its variance is rho = -0.75.

# In[95]:

v0 = 0.04
kappa = 0.1
theta = v0
sigma = 0.1
rho = -0.75


# In[96]:

spot_handle = ql.QuoteHandle(spot)
risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(evaluationDate, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(evaluationDate, dividend_rate, day_count)) 


# Construct the Heston model

# In[97]:

heston_process = ql.HestonProcess(risk_free_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)
heston_model = ql.HestonModel(heston_process)


# Lets compute the theoretical price using the AnalyticEuropeanEngine

# In[98]:

engine = ql.AnalyticHestonEngine(heston_model)
option.setPricingEngine(engine)


# In[100]:

option.NPV()


# Create put option

# In[101]:

option_type_new = ql.Option.Put
strike_new = 130
maturity_date_new = ql.Date(15, 1, 2016)

payoff_new = ql.PlainVanillaPayoff(option_type_new, strike_new)
exercise_new = ql.EuropeanExercise(maturity_date_new)
option_new = ql.VanillaOption(payoff_new, exercise_new)

option_new.setPricingEngine(engine)


# Make plot

# In[102]:

xs = np.linspace(10, 240.0, 100)
ys = []
ys2 = []
for x in xs:
    spot.setValue(x)
    ys.append(option.NPV())
    ys2.append(option_new.NPV())
    
plt.figure(figsize=(15, 6), dpi=80)
plt.subplot(1, 1, 1)
plt.plot(xs, ys, color="blue", linewidth=1.0, linestyle="-", label = "Call")
plt.plot(xs, ys2, color="green", linewidth=1.0, linestyle="-", label = "Put")
plt.legend(loc='upper center')    


# ---

# # Modeling Vanilla Interest Rate Swaps 

# An Interest Rate Swap is a financial derivative instrument in which two parties agree to exchange interest rate cash flows based on a notional amount from a fixed rate to a floating rate or from one floating rate to another floating rate.
# 
# Here we will consider an example of a plain vanilla USD swap with 10 million notional and 10 year maturity. Let the fixed leg pay 2.5% coupon semiannually, and the floating leg pay Libor 3m quarterly.

# Construct discount curve and libor curve

# In[103]:

risk_free_rate = 0.01
libor_rate = 0.02
day_count = ql.Actual365Fixed()

discount_curve = ql.YieldTermStructureHandle(ql.FlatForward(evaluationDate, risk_free_rate, day_count))
libor_curve = ql.YieldTermStructureHandle(ql.FlatForward(evaluationDate, libor_rate, day_count))

libor3M_index = ql.USDLibor(ql.Period(3, ql.Months), libor_curve)


# To construct the Swap instrument, we have to specify the fixed rate leg and floating rate leg. We construct the fixed rate and floating rate leg schedules below.

# In[107]:

calendar = ql.UnitedStates()
settle_date = calendar.advance(evaluationDate, 5, ql.Days)
maturity_date = calendar.advance(settle_date, 10, ql.Years)

fixed_leg_tenor = ql.Period(6, ql.Months)
fixed_schedule = ql.Schedule(settle_date, 
                          maturity_date, 
                          fixed_leg_tenor,
                          calendar,
                          ql.ModifiedFollowing,
                          ql.ModifiedFollowing,
                          ql.DateGeneration.Forward, 
                          False)

float_leg_tenor = ql.Period(3, ql.Months)
float_schedule = ql.Schedule(settle_date,
                          maturity_date,
                          float_leg_tenor,
                          calendar,
                          ql.ModifiedFollowing, 
                          ql.ModifiedFollowing,
                          ql.DateGeneration.Forward,
                          False)


# Below, we construct a VanillaSwap object by including the fixed and float leg schedules created above.

# In[108]:

notional = 10000000
fixed_rate = 0.025
fixed_leg_daycount = ql.Actual360()
float_leg_daycount = ql.Actual360()
float_spread = 0.004

ir_swap = ql.VanillaSwap(ql.VanillaSwap.Payer,
                      notional,
                      fixed_schedule,
                      fixed_rate,
                      fixed_leg_daycount,
                      float_schedule,
                      libor3M_index,
                      float_spread,
                      float_leg_daycount)


# We evaluate the swap using a discounting engine.

# In[109]:

swap_engine = ql.DiscountingSwapEngine(discount_curve)
ir_swap.setPricingEngine(swap_engine)


# #### Result Analysis

# The cashflows for the fixed and floating leg can be extracted from the ir_swap object. The fixed leg cashflows are shown below:

# In[110]:

for i, cf in enumerate(ir_swap.leg(0)):
    print ("%2d  %-20s  %10.2f" % (i+1, cf.date(), cf.amount()))


# In[111]:

for i, cf in enumerate(ir_swap.leg(1)):
    print ("%2d  %-20s  %10.2f" % (i+1, cf.date(), cf.amount()))


# Some other analytics such as the fair value, fair spread etc can be extracted as shown below.

# In[113]:

print ("%-20s: %20.3f" % ("Net Present Value", ir_swap.NPV()))
print ("%-20s: %20.3f" % ("Fair Spread", ir_swap.fairSpread()))
print ("%-20s: %20.3f" % ("Fair Rate", ir_swap.fairRate()))
print ("%-20s: %20.3f" % ("Fixed Leg BPS", ir_swap.fixedLegBPS()))
print ("%-20s: %20.3f" % ("Floating Leg BPS", ir_swap.floatingLegBPS()))


# ---

# # Nelson-Siegel model

# In[157]:

calendar = ql.TARGET()
today = evaluationDate


# Create bonds

# In[158]:

number_of_bonds = 15
faceValue = 100.0

quotes_handle = []    
for i in range(number_of_bonds):
    quotes_handle.append(ql.QuoteHandle(ql.SimpleQuote(faceValue)))

coupons_real = [0.1005, 0.0995, 0.0968, 0.0929, 0.0927,
                0.0927, 0.0918, 0.0914, 0.0912, 0.0911,
                0.0915, 0.0912, 0.0909, 0.0903, 0.0907]

tenor = ql.Period(3, ql.Months)
day_counter = ql.Thirty360()
accrualConvention = ql.Unadjusted
convention = ql.Unadjusted

bondSettlementDays = 0
curveSettlementDays = 0
bondSettlementDate = calendar.advance(today, bondSettlementDays, ql.Days)


# Create helpers

# In[159]:

market_price = faceValue
bonds_yields = []
instrumentsA = []
maturities = []

for j in range(number_of_bonds):
    
    # bonds every three months
    maturity = bondSettlementDate + (j+1)*tenor
    maturities.append(str(maturity))
   
    
    schedule = ql.Schedule(bondSettlementDate, 
                           maturity, 
                           tenor,
                           calendar,
                           accrualConvention,
                           accrualConvention,
                           ql.DateGeneration.Forward,
                           False)
    
    fixedRateBond = ql.FixedRateBond(bondSettlementDays, faceValue, schedule, [coupons_real[j]], day_counter)
    bond_yield = fixedRateBond.bondYield(market_price, day_counter, ql.Continuous, ql.Quarterly)
    bonds_yields.append(bond_yield)
    
    helperA = ql.FixedRateBondHelper(quotes_handle[j],
                                     bondSettlementDays,
                                     faceValue,
                                     schedule,
                                     [coupons_real[j]], 
                                     day_counter,
                                     convention)     
    instrumentsA.append(helperA)


# In[170]:

"""
cash_flows = fixedRateBond.cashflows()
def price_error_given_yield(rate):
    interestRate = ql.InterestRate(rate, dayCount, compounding, compoundingFrequency)    
    return ql.CashFlows_npv(cash_flows, interestRate, False) - market_price     

solver = ql.Bisection()
accuracy = 0.000001
solver_max = 0.15
solver_min = 0.0025
irr = solver.solve(price_error_given_yield, accuracy, solver_min, solver_max)
irr

"""


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# Set parameters for optimization

# In[160]:

constrainAtZero = True
tolerance = 1.0e-5;
max_iterations = 5000;


# There are some fitting algorithms

# In[161]:

nelsonSiegel = ql.NelsonSiegelFitting()
svensson = ql.SvenssonFitting()
simplePolynomial = ql.SimplePolynomialFitting(3, constrainAtZero)
exponentialSplines = ql.ExponentialSplinesFitting(constrainAtZero)


# In[162]:

term_structure = ql.FittedBondDiscountCurve(curveSettlementDays, 
                                 calendar, 
                                 instrumentsA,
                                 dayCount,
                                 nelsonSiegel,
                                 tolerance,
                                 max_iterations)


# Fitted parameters chosen model

# In[163]:

a = term_structure.fitResults()
b = a.solution()
for i in range(len(b)):
    print(b[i])


# In[164]:

# This method requires for cut day suffixes
import re
def solve(s):
    ss =  re.sub(r'(\d)(st|nd|rd|th)', r'\1', s)
    ss = ss.replace(',', '')
    return ss


# Getting curve

# In[165]:

spots_model = []
dates_model = []
i = -1

while True:    
    
    i+=1
    date = evaluationDate + ql.Period(i, ql.Days)
    
    if date > term_structure.maxDate() - ql.Period(1, ql.Years):        
        break
     
    dates_model.append(str(date))
    yrs = dayCount.yearFraction(evaluationDate, date)
    compounding = ql.Continuous
    freq = ql.Quarterly
    zero_rate = term_structure.zeroRate(yrs, compounding, freq)   
    eq_rate = zero_rate.equivalentRate(day_counter, compounding, freq, evaluationDate,date).rate()
    spots_model.append(zero_rate.rate())


# Prepare dateTime axis

# In[166]:

from datetime import datetime

dates_by_model = []

for i in range(len(dates_model)):
    dates_points = datetime.strptime(solve(dates_model[i]), '%B %d %Y')
    dates_by_model.append(dates_points)

dates_by_points = []

for i in range(len(maturities)):
    dates_points = datetime.strptime(solve(maturities[i]), '%B %d %Y')
    dates_by_points.append(dates_points)
    
dates_by_model_graph = matplotlib.dates.date2num(dates_by_model)
dates_by_points_graph = matplotlib.dates.date2num(dates_by_points)    


# In[ ]:




# Plot

# In[167]:

plt.figure(figsize=(17, 10), dpi=80)
plt.plot_date(dates_by_model_graph, spots_model, "b-", lw=1, alpha=0.6)
plt.plot_date(dates_by_points_graph, coupons_real)


# In[168]:

plt.figure(figsize=(17, 10), dpi=80)
plt.plot_date(dates_by_model_graph, spots_model, "b-", lw=1, alpha=0.6)
plt.plot_date(dates_by_points_graph, bonds_yields)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ---

# # Stochastic straying

# In[133]:

bmp = ql.GeometricBrownianMotionProcess(1.0, 1.0, 1.0)


# In[134]:

rnd = ql.UniformRandomGenerator()


# In[135]:

steps = 1000


# In[136]:

len_time = 1


# In[137]:

seq = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(steps, rnd))


# In[138]:

pahtGen = ql.GaussianPathGenerator(bmp, len_time, steps, seq, True)


# In[151]:

num_paths = 3


# In[152]:

time = []
paths = np.zeros((num_paths, steps+1))
for i in range(num_paths):
    path = pahtGen.next().value()
    time = [path.time(j) for j in range(len(path))]   
    value = [path[j] for j in range(len(path))]
    paths[i, :] = np.array(path)
    
time = np.array(time)    


# In[153]:

for i in range(num_paths):
    plt.plot(time, paths[i, :], lw=0.8, alpha=0.6)
plt.title("Random")
plt.show()


# In[100]:




# In[524]:




# In[9]:




# In[ ]:




# In[2]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



