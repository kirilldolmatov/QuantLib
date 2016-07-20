
# coding: utf-8

# In[1]:

from datetime import datetime, date, time
import QuantLib as ql


# ### Internal Objects

# In[2]:

class Bond(object):
    
    def __init__(self, issuer, bond_name, price, tenor, face_amount):
        self.dates = []
        self.cashflows = [] 
        self.issuer = issuer
        self.bond_name = bond_name
        self.price = ql.QuoteHandle(ql.SimpleQuote(price))        
        self.tenor = tenor
        self.face_amount = face_amount
                
    def add_date(self, date):
        day, month, year = map(int, date.split('.'))
        self.dates.append(ql.Date(day, month, year))
    
    def add_cashflow(self, cashflow):
        self.cashflows.append(float(cashflow))
    


# ### Import Data

# In[3]:

face_amount = 1000.0 # for all bonds face_amount = 1000.0
tenor = ql.Period(6, ql.Months) # for all bonds tenor = 1000.0

bonds = {}

with open('bonds.txt') as f:
    next(f) #skip header
    
    for line in f:
        s = line.rstrip().split(';')
        bond_name = s[1]
        if bond_name not in bonds:
            issuer = s[0]        
            price = float(s[4])            
            bonds[bond_name] = Bond(issuer, bond_name, price, tenor, face_amount)
        
        bonds[bond_name].add_date(s[2])
        bonds[bond_name].add_cashflow(s[3])


# ### Set QuantLib Param

# In[4]:

evaluationDate = ql.Date(1, 6, 2016)
ql.Settings.instance().evaluationDate = evaluationDate
calendar = ql.TARGET()
day_counter = ql.Thirty360()
accrualConvention = ql.Unadjusted
bussiness_convention = ql.Unadjusted
bondSettlementDays = 0
curveSettlementDays = 0
bondSettlementDate = calendar.advance(evaluationDate, bondSettlementDays, ql.Days)


# ### Make QuantLib Packaging

# In[5]:

instruments = []
instruments_names = []

for bond in bonds.keys():
    
        schedule = ql.Schedule(bonds[bond].dates[0] - bonds[bond].tenor, 
                               bonds[bond].dates[-1], 
                               bonds[bond].tenor,
                               calendar,
                               accrualConvention,
                               accrualConvention,
                               ql.DateGeneration.Forward,
                               False)
 
        helperA = ql.FixedRateBondHelper(bonds[bond].price,
                                         bondSettlementDays,
                                         bonds[bond].face_amount,
                                         schedule,
                                         bonds[bond].cashflows,
                                         day_counter,
                                         bussiness_convention)
           
        instruments.append(helperA)
        instruments_names.append(bond)


# ### Check Cashflows

# In[7]:

for index, instrument in enumerate(instruments):    
    for cashflow in instrument.bond().cashflows():
        print('{!s:15} {!s:20} {!s:20}'.format(instruments_names[index], cashflow.date(), cashflow.amount()))
    print('\t')    


# ### Make QuantLib Optimization

# In[8]:

tolerance = 1.0e-5
iterations = 50000
nelsonSiegel = ql.NelsonSiegelFitting()
term_structure = ql.FittedBondDiscountCurve(curveSettlementDays, 
                                 calendar, 
                                 instruments,
                                 day_counter,
                                 nelsonSiegel,
                                 tolerance,
                                 iterations)
a = term_structure.fitResults()

