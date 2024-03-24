# __author__ = 'koushik'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import dateutils as du
import sys
import pickle as pk
import matplotlib.pyplot as plt
import quandl as quandl
import scipy as sp

class const():
    @staticmethod
    def loanIndxMax():
        return 100000
    @staticmethod
    def maxTerm():
        return 61
    @staticmethod
    def pi():
        return 3.14159
    @staticmethod
    def ageKnots():
        return 5
    @staticmethod
    def maxTrees():
        return 100
    @staticmethod
    def maxDepth():
        return None
    @staticmethod
    def maxLeaf():
        return 1
    @staticmethod
    def chartFields():
        return ['term','MOB','orig_fico','vintage','month','loan_amnt','coupon','purpose','emp_length','region','y_mod','y_act']
    @staticmethod
    def charFields():
        return ['id','annual_inc','total_acc','open_acc','loan_amnt','addr_state','purpose']
    @staticmethod
    def pmtDropFields():
        return ['CO','State','MonthlyIncome','EarliestCREDITLine','OpenCREDITLines',
                'TotalCREDITLines','RevolvingCREDITBalance','MonthsSinceDQ','PublicRec','MonthsSinceLastRec',
                'currentpolicy','Last_FICO_BAND','VINTAGE','RECEIVED_D']
    @staticmethod
    def regressorFields():
        return ['age_k0','age_k1','age_k2','age_k3','age_k4','age_k5','age_k6','age_k7','age_k8','age_k9','nevlate','term60',
                'age_k0.term36','age_k1.term36','age_k2.term36','age_k3.term36','age_k4.term36','age_k5.term36',
                'age_k6.term36','age_k7.term36','age_k8.term36','age_k9.term36',
                'age_k0.term60','age_k1.term60','age_k2.term60','age_k3.term60','age_k4.term60','age_k5.term60',
                'age_k6.term60','age_k7.term60','age_k8.term60','age_k9.term60',
                'dti','fico675','fico700','fico750','fico800',
                'vint2009','vint2010','vint2011','vint2012','vint2013','vint2014','vint2015',
                'm2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12',
                'ls0','ls5','ls10','ls15','ls20','ls25','coup5','coup7.5','coup10','coup12.5','coup15','coup17.5',
                'lp_debt','lp_cc','lp_hi','lp_mp','el<1','el1','el2','el3','el4','el5','el6','el7','el8','el10+','elna',
                'inql6m','anninc','totacc','revutil','del2y','openacc','home_own','regSouth','regNE','RegPac','RegMW']
    @staticmethod
    def startStates():
        return ['C','D3','D6','D6+']
    @staticmethod
    def endStates():
        return ['C','D3','D6','D6+','D','P']
    @staticmethod
    def allTransitions():
        return ['CtoC', 'CtoD3', 'CtoD6', 'CtoD6+', 'CtoD', 'CtoP',
               'D3toC', 'D3toD3', 'D3toD6', 'D3toD6+', 'D3toD', 'D3toP',
               'D6toC', 'D6toD3', 'D6toD6', 'D6toD6+', 'D6toD', 'D6toP',
               'D6+toC', 'D6+toD3', 'D6+toD6', 'D6+toD6+', 'D6+toD', 'D6+toP']
    @staticmethod
    def exemptTransitions():
        return ['CtoD6','CtoD6+','CtoD','D3toD6+','D6toP','D6+toD3','D6+toP']
    @staticmethod
    def curveHeaders():
        return ['NewOrig','BegBal','PrinPaid','PrinPrepaid','Defs','EndBal','IntPaid','CPR','CDR']
    @staticmethod
    def priceHeaders():
        return ['LoanID','MOB','Term','Status','Amount','Coupon','ParSpread','ParYield','ModYield','AvgYield','ActYield','ModPrice','AvgPrice','ActPrice']
    @staticmethod
    def cumulHeaders():
        return ['CPR_mod','CPR_act','CDR_mod','CDR_act']
    @staticmethod
    def quandlKey():
        return 'oJQ8bM4ArnvxyLUSNjgf'

def read_pmt_data(pmtStr):
# Purpose: Reads in pmts and char files

    dtPmts = pd.read_csv(pmtStr, header=0, index_col=False)
    dtPmts = dtPmts.sort_values(by=['LOAN_ID','MOB'])
    dtPmts = dtPmts.reset_index(drop=True)
    dtPmts = dtPmts.drop(const.pmtDropFields(),axis=1)
    print('\nReading in LC payments data...')

    return dtPmts

def read_stats_data(dtPmts):
# Purpose: Reads in stats or characteristics file

    dtChars = pd.DataFrame()
    charFiles = ['3a.csv', '3b.csv', '3c.csv', '3d.csv', '2016Q1.csv', '2016Q2.csv','2016Q3.csv']
    for c in charFiles:
        print('Reading in file: %s...' % c)
        dtChars = pd.concat([dtChars, pd.read_csv('LC Data/' + c, header=1, index_col=False)])

    #dtChars.to_csv('Loan Stats.csv')
    dtChars = dtChars[const.charFields()]
    #dropIndx = np.where(pd.isnull(dtChars['purpose']))
    #dtChars = dtChars.drop(dtChars.index[dropIndx])

    dtPmts = pd.merge(dtPmts, dtChars, how='inner', left_on='LOAN_ID', right_on='id', copy=False)
    dtPmts = dtPmts.drop('id',axis=1)

    print('Merging in stats purpose ...')

    return dtPmts

def read_dict(path):
# Purpose: Reads in dict from the path.  File contains lookups to append to dataframe.

    dtD = pd.read_csv('Inputs/'+path, header=0, index_col=False)
    print('Read in file from path: %s ...' %path)
    return dtD

def read_rates():

    dtR = quandl.get('FED/SVENY', rows=1).ix[:,:5]
    fRates = sp.interpolate.interp1d([0,12,24,36,48,60],np.insert(dtR.values,0,0),kind='cubic')

    return fRates(range(1,const.maxTerm()))

def pickle_save(dtPmts,saveStr):
# Purpose: Stores pandas dataframe as a pickled .p file

    pk.dump(dtPmts, open('Pickled/'+saveStr, 'wb'))
    print('Pickled file ...')

def pickle_load(loadStr):
# Purpose: Loads object from a pickled .p file

    dtOut = pd.DataFrame()

    for l in loadStr:
        print('Loaded pickled file from: %s...' %l)
        dtOut = pd.concat([dtOut,pk.load(open('Pickled/'+l, 'rb'))],axis=0)

    return dtOut


def clean_pmt_data(dtPmts):
# Purpose: Cleans data from LC pmts file so inter- and intra-temporal pmts word

    # Fix: Round numerical columns
    roundCols = ['LOAN_ID', 'PBAL_BEG_PERIOD', 'PRNCP_PAID', 'INT_PAID', 'FEE_PAID', 'DUE_AMT',
                 'RECEIVED_AMT', 'PBAL_END_PERIOD','COAMT']

    for r in roundCols:
        dtPmts[r] = ((100*dtPmts[r]).round(0))/100
    print('\nFix: Rounded numerical columns ...')

    # Fix: Convert month dates to datetimes
    dtPmts['MONTH'] = pd.to_datetime(dtPmts['MONTH'],format='%b%Y')
    dtPmts['IssuedDate'] = pd.to_datetime(dtPmts['IssuedDate'], format='%b%Y')
    dtPmts['inception_year'] = dtPmts['IssuedDate'].dt.year
    dtPmts = dtPmts.iloc[np.where(np.logical_and(dtPmts['inception_year'] > 2011,dtPmts['inception_year'] < 2014))]
    print('\nFix: Converted dates to datetimes ...')

    # Fix: Convert PCO's from NaN to 0
    dtPmts['PCO_RECOVERY'] = dtPmts['PCO_RECOVERY'].fillna(0)
    dtPmts['PCO_COLLECTION_FEE'] = dtPmts['PCO_COLLECTION_FEE'].fillna(0)
    dtPmts['grade'] = dtPmts['grade'].fillna('NR')
    dtPmts['MONTH'] = dtPmts['MONTH'].fillna('3000-13-32')
    dtPmts['RevolvingLineUtilization'] = dtPmts['RevolvingLineUtilization'].fillna(-1)
    #dtPmts['PBAL_BEG_PERIOD'] = dtPmts['PBAL_BEG_PERIOD'].fillna(0)
    #dtPmts['PBAL_END_PERIOD'] = dtPmts['PBAL_END_PERIOD'].fillna(0)
    #dtPmts['PRNCP_PAID'] = dtPmts['PRNCP_PAID'].fillna(0)
    #dtPmts['COAMT'] = dtPmts['COAMT'].fillna(0)
    print('\nFix: Removed NaNs from PCO, grade, month, balances, payments fields ...')

    # Fix: Average out FICO's
    dtPmts['fico'] = 0.5 * (dtPmts['APPL_FICO_BAND'].apply(lambda x: float(x[:3])) +
                             dtPmts['APPL_FICO_BAND'].apply(lambda x: float(x[-3:])))
    dtPmts = dtPmts.drop('APPL_FICO_BAND',axis=1)
    print('\nFix: Created avg fico column ...')

    # Fix: Make End Bal = 0, when charge off occurs
    coIndx = np.where(dtPmts['COAMT'] > 0)
    dtPmts['PBAL_END_PERIOD'].iloc[coIndx] = 0
    print('\nFix: Zeroed out end bal when loans charged off...')

    # Fix: Find non-consecutive pmt months and insert a row
    pmtIndx = np.where(np.logical_and((12 * dtPmts['MONTH'].iloc[1:].dt.year + dtPmts['MONTH'].iloc[1:].dt.month).values -
                                      (12 * dtPmts['MONTH'].iloc[:-1].dt.year + dtPmts['MONTH'].iloc[:-1].dt.month).values == 2,
                                      dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))
    dtNew = dtPmts.iloc[pmtIndx].copy(deep=True)
    dtNew[['PRNCP_PAID','INT_PAID','FEE_PAID','RECEIVED_AMT','DUE_AMT','COAMT']] = 0
    dtNew['PBAL_BEG_PERIOD'] = dtNew['PBAL_END_PERIOD']
    dtNew['MONTH'] = dtNew['MONTH'].apply(lambda x: x + du.relativedelta(months=1))
    dtPmts = pd.concat([dtPmts,dtNew],axis=0)

    dtPmts = dtPmts.sort_values(['LOAN_ID','MONTH'],ascending=[True,True])
    dtPmts = dtPmts.reset_index(drop=True)
    print('\nFix: Inserted zero row for non-consecutive payments ...')

    # Fix: Find multiple payments in one month and delete a row
    pmtIndx = np.where(np.logical_and(dtPmts['MONTH'].iloc[1:].values == dtPmts['MONTH'].iloc[:-1].values,
                                        dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))[0]

    dtPmts[['PRNCP_PAID','INT_PAID','FEE_PAID','RECEIVED_AMT','DUE_AMT','COAMT']].iloc[pmtIndx] += dtPmts[['PRNCP_PAID','INT_PAID','FEE_PAID','RECEIVED_AMT','DUE_AMT','COAMT']].iloc[pmtIndx+1]
    dtPmts['PBAL_END_PERIOD'].iloc[pmtIndx] = dtPmts['PBAL_END_PERIOD'].iloc[pmtIndx+1].values
    dtPmts = dtPmts.drop(dtPmts.index[pmtIndx+1])

    dtPmts = dtPmts.sort_values(['LOAN_ID','MONTH'],ascending=[True,True])
    dtPmts = dtPmts.reset_index(drop=True)
    print('\nFix: Consolidated payments in same month for same loan ...')

    # Fix: Ensure a zero payment exists when difference between MOB 1 and the issue date

    # Fix: Insert zero payment for most recent month and last month all active loans

    lastMonth = np.sort(dtPmts['MONTH'].unique())[-1]
    prevMonth = np.sort(dtPmts['MONTH'].unique())[-2]
    prevpmtIndx = np.where(np.logical_and(dtPmts['MONTH'] == prevMonth,dtPmts['PBAL_END_PERIOD'] > 1))
    lastpmtIndx = np.where(dtPmts['MONTH'] == lastMonth)

    loanIndx = dtPmts['LOAN_ID'].iloc[prevpmtIndx].unique()
    loanIndx = loanIndx[np.where(np.in1d(loanIndx,dtPmts['LOAN_ID'].iloc[lastpmtIndx]) == False)]
    prevpmtIndx = prevpmtIndx[0][np.in1d(dtPmts['LOAN_ID'].iloc[prevpmtIndx],loanIndx)]

    dtNew = dtPmts.iloc[prevpmtIndx].copy(deep=True)
    dtNew[['PRNCP_PAID','INT_PAID','FEE_PAID','RECEIVED_AMT','DUE_AMT','COAMT']] = 0
    dtNew['PBAL_BEG_PERIOD'] = dtNew['PBAL_END_PERIOD']
    dtNew['MONTH'] = lastMonth
    dtPmts = pd.concat([dtPmts,dtNew],axis=0)

    dtPmts = dtPmts.sort_values(['LOAN_ID','MONTH'],ascending=[True,True])
    dtPmts = dtPmts.reset_index(drop=True)
    print('\nFix: Added zero rows for loans missing last reported months data ...')

    # Fix: Check to ensure AMT PAID = INT PAID + FEE PAID, if not modify

    pmtIndx = np.where(dtPmts['RECEIVED_AMT'] - dtPmts['PRNCP_PAID'] - dtPmts['INT_PAID'] - dtPmts['FEE_PAID'] != 0)
    dtPmts['RECEIVED_AMT'].iloc[pmtIndx] = dtPmts['PRNCP_PAID'].iloc[pmtIndx] + dtPmts['INT_PAID'].iloc[pmtIndx] + dtPmts['FEE_PAID'].iloc[pmtIndx]
    print('\nFix: Added up received amount ...')

    # Fix: Make sure charge offs only occur once
    pmtIndx = np.where(np.logical_and(dtPmts['PERIOD_END_LSTAT'].iloc[:-1] == 'Charged Off',
                                     dtPmts['PERIOD_END_LSTAT'].iloc[1:] == 'Charged Off'))
    dtPmts = dtPmts.drop(dtPmts.index[pmtIndx[0]+1])

    dtPmts = dtPmts.sort_values(['LOAN_ID','MONTH'],ascending=[True,True])
    dtPmts = dtPmts.reset_index(drop=True)

    print('\nFix: Dropped multiple charged off statuses ...')

    # Fix: Find where next opening balance is greater than previous opening balance for same loan
    pmtIndx = np.where(np.logical_and(dtPmts['PBAL_BEG_PERIOD'].iloc[1:].values > dtPmts['PBAL_BEG_PERIOD'].iloc[:-1].values,
                                      dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))
    loanIndx = np.unique(dtPmts['LOAN_ID'].iloc[pmtIndx])
    for l in loanIndx:
    #l = loanIndx[14]
        rIndx = np.where(dtPmts['LOAN_ID'] == l)
        for r in range(0,len(rIndx[0])-1):
            if (dtPmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r] > dtPmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r-1]):
                dtPmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r-1] = dtPmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r]

    print('\nFix: Made sure opening bal is decreasing over time ...')

    # Fix: Find loans where next opening balance does not match previous closing balance

    pmtIndx = np.where(np.logical_and(dtPmts['PBAL_BEG_PERIOD'].iloc[1:].values != dtPmts['PBAL_END_PERIOD'].iloc[:-1].values,
                                    dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))
    dtPmts['PBAL_END_PERIOD'].iloc[pmtIndx[0]] = dtPmts['PBAL_BEG_PERIOD'].iloc[pmtIndx[0]+1].values

    print('\nFix: Made sure consecutive opening and closing bals match ...')

    # Fix: Check to ensure PBAL BEG - PRIN PAID - COAMT = PBAL END, if not modify PRIN PAID

    pmtIndx = np.where(dtPmts['PBAL_BEG_PERIOD']-dtPmts['PRNCP_PAID']-dtPmts['COAMT']-dtPmts['PBAL_END_PERIOD'] != 0)
    dtPmts['PRNCP_PAID'].iloc[pmtIndx] = dtPmts['PBAL_BEG_PERIOD'].iloc[pmtIndx] - dtPmts['COAMT'].iloc[pmtIndx] - dtPmts['PBAL_END_PERIOD'].iloc[pmtIndx]
    print('\nFix: Reconciled balances and payments ...')

    # Fix: Renumber MOB's

    startIndx = np.where(dtPmts['LOAN_ID'].iloc[:-1].values != dtPmts['LOAN_ID'].iloc[1:].values)[0] + 1
    startIndx = np.insert(startIndx,0,0)
    subIndx = pd.DataFrame(data=np.arange(0,dtPmts.shape[0]))
    subIndx.iloc[np.where(np.in1d(subIndx,startIndx)==False)[0]] = np.nan
    subIndx = subIndx.ffill()
    dtPmts['MOB'] = np.subtract(np.arange(0,dtPmts.shape[0]),subIndx.values.transpose()).transpose() + 1

    print('\nFix: Added MOBs ...')

    # Fix: Add unscheduled and principals paid
    dtPmts['unscheduled_principal'] = np.maximum(dtPmts['RECEIVED_AMT'] - dtPmts['DUE_AMT'], 0)
    dtPmts['scheduled_principal'] = dtPmts['PRNCP_PAID'] - dtPmts['unscheduled_principal']

    print('\nFix: Added unscheduled and scheduled principal paid ...')

    # Fix: Add new origination amount
    dtPmts['new_orig'] = 0
    newOrigIndx = np.where(dtPmts['MOB'] == 1)
    dtPmts['new_orig'].iloc[newOrigIndx] = dtPmts['PBAL_BEG_PERIOD'].iloc[newOrigIndx].values
    print('\nFix: Added new originations ...')

    # replace when add check for last month in clean_data
    #dropIndx = np.where(dtPmts['MONTH'] == '2016-12-01')
    #dtPmts = dtPmts.drop(dtPmts.index[dropIndx])
    #dtPmts = dtPmts.reset_index(drop=True)

    return dtPmts

def check_pmt_data(dtPmts):
# Purpose: Checks to ensure inter-, intra-temporal pmts, and MOBs are aligned

    # Check 1: Make sure walk works
    walkIndx = np.where(np.logical_and(np.abs(np.subtract(dtPmts['PBAL_BEG_PERIOD'].iloc[1:].values,dtPmts['PBAL_END_PERIOD'].iloc[:-1].values)) > 1,
                     dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))

    if len(walkIndx[0]) == 0:
        print('\nWalk check works...')
    else:
        print('\nProblem with walk check, but fixed it...')
        dropIndx = np.where(np.in1d(dtPmts['LOAN_ID'],np.unique(dtPmts['LOAN_ID'].iloc[walkIndx])))
        dtPmts = dtPmts.drop(dtPmts.index[dropIndx])

    # Check 2: Make sure months are consecutive
    monthIndx = np.where(np.logical_and.reduce((dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values,
                    pd.DatetimeIndex(dtPmts['MONTH'].iloc[:-1]) + pd.offsets.MonthBegin(1) != pd.DatetimeIndex(dtPmts['MONTH'].iloc[1:]),
                    dtPmts['PERIOD_END_LSTAT'].iloc[1:] != 'Charged Off')))

    if len(monthIndx[0]) == 0:
        print('\nMonth check works...')
    else:
        print('\nProblem with month check, but fixed it ...')
        dropIndx = np.where(np.in1d(dtPmts['LOAN_ID'], np.unique(dtPmts['LOAN_ID'].iloc[monthIndx])))
        dtPmts = dtPmts.drop(dtPmts.index[dropIndx])

    # Check 3: Make sure MOBs are consecutive
    mobIndx = np.where(np.logical_and.reduce((dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values,
                    np.subtract(dtPmts['MOB'].iloc[:-1].values,dtPmts['MOB'].iloc[1:].values) > 1)))

    if len(mobIndx[0]) == 0:
        print('\nMOB check works...')
    else:
        print('\nProblem with MOB check, but fixed it...')
        dropIndx = np.where(np.in1d(dtPmts['LOAN_ID'], np.unique(dtPmts['LOAN_ID'].iloc[mobIndx])))
        dtPmts = dtPmts.drop(dtPmts.index[dropIndx])

    return dtPmts

def insert_status(dtPmts):
# Purpose: Inserts status number as column

    loan_status = np.zeros(shape=(dtPmts.shape[0],1))

    loan_status[:,0] = np.round((dtPmts['DUE_AMT'].values - dtPmts['RECEIVED_AMT'].values)/dtPmts['MONTHLYCONTRACTAMT'].values,
                           decimals=0) # calculate number of pmts missed (not based on time)
    loan_status[np.where(loan_status > 3),0] = 3 # max amount missed is 3 payment cycles

    loan_status[np.where(np.logical_or(dtPmts['PERIOD_END_LSTAT'] == 'Current', # current: status = current or rec amt > due amt
                                       dtPmts['RECEIVED_AMT'] >= dtPmts['DUE_AMT'])),0] = 0

    loan_status[np.where(dtPmts['COAMT'] > 0),0] = 4 # defaulted or charged off

    #loan_status[np.where(np.logical_or(dtPmts['PERIOD_END_LSTAT'] == 'Default', # defaulted or charged off
    #                                   dtPmts['PERIOD_END_LSTAT'] == 'Charged Off')),0] = 4

    loan_status[np.where(np.logical_and(dtPmts['PERIOD_END_LSTAT'] == 'Fully Paid', # prepaid: fully paid before term is up
                                       dtPmts['MOB'] <= dtPmts['term'])),0] = 5

    loan_status[np.where(np.logical_or(dtPmts['PERIOD_END_LSTAT'] == 'In Grace Period',  # other
                                       dtPmts['PERIOD_END_LSTAT'] == 'Issued')), 0] = -1

    #loan_status = pd.DataFrame(data=loan_status,index=dtPmts.index,columns=['loan_status'])
    dtPmts['loan_status'] = loan_status

    print('\nFinished adding status column...')

    return dtPmts

def insert_delq_indicator(dtPmts):
# Purpose: To be deprecated

    firstDelqStatus = np.zeros(shape=(dtPmts.shape[0], 1))
    secondDelqStatus = np.zeros(shape=(dtPmts.shape[0], 1))

    firstDelqStatus[np.in1d(dtPmts['LOAN_ID'],dtPmts['LOAN_ID'].iloc[np.where(dtPmts['loan_status'] == 1)].unique()),0] = 1
    secondDelqStatus[np.in1d(dtPmts['LOAN_ID'],dtPmts['LOAN_ID'].iloc[np.where(dtPmts['loan_status'] == 2)].unique()),0] = 1

    dtPmts['first_delq'] = firstDelqStatus
    dtPmts['second_delq'] = secondDelqStatus

    return dtPmts

def pre_processing(dtPmts):
# Purpose: Shifts certain column formats

    #dtPmts['RevolvingLineUtilization'] = dtPmts['RevolvingLineUtilization'].str.rstrip('%').astype('float64').fillna(-100) / 100
    dtPmts['loan_month'] = dtPmts['MONTH'].dt.month
    print('Added loan_month ... ')

    dtPmts['coupon'] = dtPmts['InterestRate'] * 100
    print('Added coupon ... ')

    lateIndx = np.where(np.logical_and(dtPmts['loan_status'] > 0,dtPmts['loan_status'] < 5))[0]
    endIndx = np.append(np.where(dtPmts['MOB'] == 1)[0][1:]-1,dtPmts.shape[0]-1)
    endIndx = endIndx[np.searchsorted(endIndx,lateIndx,'left')]

    lateRows = []
    for i in range(0,len(lateIndx)):
        lateRows.append(np.arange(lateIndx[i],endIndx[i]+1))
    lateRows = np.unique(np.concatenate(lateRows).ravel())

    dtPmts['never_late_indicator'] = 1
    dtPmts['never_late_indicator'].iloc[lateRows] = 0
    print('Added never_late_indicator ... ')

    # Add region
    dtRegion = read_dict('states.csv')
    dtPmts = pd.merge(dtPmts, dtRegion, how='left', left_on='addr_state', right_on='state', copy=False)
    #dtPmts = dtPmts.drop(['state'], axis=1)
    print('Added field: region ...')

    dtPmts = dtPmts.drop(['InterestRate', 'addr_state', 'state'], axis=1)

    print('\nProcessed data for LC payments and borrower data...')

    return dtPmts

def generate_regressors(dtPmts):
# Purpose: Generates classifiers from numerical data / input for random forest modeling

    # generates datatable of regressors
    dtRegr = np.zeros(shape=(dtPmts.shape[0],len(const.regressorFields())),dtype='float')

    # generates datatable of chart data
    dtCharts = pd.DataFrame(index=dtPmts.index,columns=const.chartFields())
    dtCharts[['y_mod','y_act']] = 0 #important for plotting later on

    # Intercept
    # dtRegr[:,0] = 1

    lastCol = 0
    #print('Intercept: %d' %lastCol)

    # transform age into linear spline combinations
    for i in range(0,10):
        indx = np.where(np.logical_and(dtPmts['MOB'] >= (i+1)*const.ageKnots(),dtPmts['MOB'] < (i+2)*const.ageKnots()))
        dtCharts['MOB'].iloc[indx] = (i+1) * const.ageKnots()
        dtRegr[indx,i] = np.maximum(dtPmts['MOB'].iloc[indx] - (i+1)*const.ageKnots(),0)
        print('MOB: %d' % lastCol)
        lastCol = lastCol+1
    dtCharts['MOB'].iloc[np.where(pd.isnull(dtCharts['MOB']))] = 0

    # never late indicator
    dtRegr[np.where(dtPmts['never_late_indicator']==1),lastCol] = 1
    print('Never late: %d' %lastCol)
    lastCol = lastCol+1

    # indicators for term
    dtRegr[np.where(dtPmts['term']==60),lastCol] = 1
    print('Term: %d' %lastCol)
    lastCol = lastCol+1
    dtCharts['term'] = dtPmts['term']

    # indicators for term * age linear spline for term=36
    for j in range(0,10):
        dtRegr[:,lastCol] = np.multiply(np.absolute(dtRegr[:,11]-1),dtRegr[:,j])
        print('Term 36 * age spline: %d' %lastCol)
        lastCol = lastCol+1

    # indicators for term * age linear spline for term=60
    for j in range(0,10):
        dtRegr[:,lastCol] = np.multiply(dtRegr[:,11],dtRegr[:,j])
        print('Term 60 * age spline: %d' %lastCol)
        lastCol = lastCol+1

    # dti_ex_mortgage with nan's set to 0
    dtRegr[:,lastCol] = dtPmts['dti']
    print('DTI: %d' %lastCol)
    lastCol = lastCol+1

    # fico ranges, exclude > 800
    ficoRanges = np.zeros(shape=(4,2),dtype='float')
    ficoRanges[:,0] = [0,675,700,750]
    ficoRanges[:,1] = [675,700,750,800]

    for i in range(0,len(ficoRanges)):
        dtRegr[np.where(np.logical_and(dtPmts['fico']<ficoRanges[i,1],dtPmts['fico']>=ficoRanges[i,0])),lastCol] = 1
        dtCharts['orig_fico'].iloc[np.where(np.logical_and(dtPmts['fico']<ficoRanges[i,1],dtPmts['fico']>=ficoRanges[i,0]))] = ficoRanges[i,1].astype(int)
        print('FICO range: %d' %lastCol)
        lastCol = lastCol+1

    dtCharts['orig_fico'].iloc[np.where(pd.isnull(dtCharts['orig_fico']))] = '800'

    # vintages
    #year = np.array(map(float,dtPmts['issue_d'].apply(lambda x: x[4:6])))+2000
    for i in range(2009,2016):
        dtRegr[np.where(dtPmts['inception_year']==i),lastCol] = 1
        dtCharts['vintage'].iloc[np.where(dtPmts['inception_year']==i)] = int(i)
        print('Vintage: %d' %lastCol)
        lastCol = lastCol+1

    dtCharts['vintage'].iloc[np.where(pd.isnull(dtCharts['vintage']))] = '2008'

    # month for seasonality effects
    for i in range(2,13):
        dtRegr[np.where(dtPmts['loan_month']==i),lastCol] = 1
        dtCharts['month'].iloc[np.where(dtPmts['loan_month']==i)] = int(i)
        print('Month: %d' %lastCol)
        lastCol = lastCol+1
    dtCharts['month'].iloc[np.where(pd.isnull(dtCharts['month']))] = 1

    # loan size ranges
    loanSizes = np.zeros(shape=(6,2),dtype='float')
    loanSizes[:,0] = [0,5000,10000,15000,20000,25000]
    loanSizes[:,1] = loanSizes[:,0] + 5000

    for i in range(0,len(loanSizes)):
        dtRegr[np.where(np.logical_and(dtPmts['loan_amnt']<loanSizes[i,1],dtPmts['loan_amnt']>=loanSizes[i,0])),lastCol] = 1
        dtCharts['loan_amnt'].iloc[np.where(np.logical_and(dtPmts['loan_amnt']<loanSizes[i,1],dtPmts['loan_amnt']>=loanSizes[i,0]))] = loanSizes[i,0].astype(int)
        print('Loan Size: %d' %lastCol)
        lastCol = lastCol+1
    dtCharts['loan_amnt'].iloc[np.where(pd.isnull(dtCharts['loan_amnt']))] = '30000'

    # coupon ranges
    loanCoupons = np.zeros(shape=(6,2),dtype='float')
    loanCoupons[:,0] = [5,7.5,10,12.5,15,17.5]
    loanCoupons[:,1] = loanCoupons[:,0] + 2.5

    for i in range(0,len(loanCoupons)):
        dtRegr[np.where(np.logical_and(dtPmts['coupon']<loanCoupons[i,1],dtPmts['coupon']>=loanCoupons[i,0])),lastCol] = 1
        dtCharts['coupon'].iloc[np.where(np.logical_and(dtPmts['coupon']<loanCoupons[i,1],dtPmts['coupon']>=loanCoupons[i,0]))] = loanCoupons[i,0]
        print('Coupon: %d' %lastCol)
        lastCol = lastCol+1

    dtCharts['coupon'].iloc[np.where(pd.isnull(dtCharts['coupon']))] = 20

    # loan purposes, dropped everything after major purchase
    loanPurposes = ['debt_consolidation','credit_card','home_improvement','major_purchase']
    for i in range(0,len(loanPurposes)):
        dtRegr[np.where(dtPmts['purpose']==loanPurposes[i]),lastCol] = 1
        dtCharts['purpose'].iloc[np.where(dtPmts['purpose']==loanPurposes[i])] = loanPurposes[i]
        print('Loan purpose: %d' %lastCol)
        lastCol = lastCol+1

    dtCharts['purpose'].iloc[np.where(pd.isnull(dtCharts['purpose']))] = 'other'

    # employment length, dropped 9 years
    empLengths = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years']
    for i in range(0,len(empLengths)):
        dtRegr[np.where(dtPmts['EmploymentLength']==empLengths[i]),lastCol] = 1
        dtCharts['emp_length'].iloc[np.where(dtPmts['EmploymentLength']==empLengths[i])] = i
        print('Employment length: %d' %lastCol)
        lastCol = lastCol+1

    dtCharts['emp_length'].iloc[np.where(pd.isnull(dtCharts['emp_length']))] = i + 1

    # inquiries in last 6m, convert nan's to 0's
    dtRegr[:,lastCol] = dtPmts['Inquiries6M']
    print('Inq last 6m: %d' %lastCol)
    lastCol = lastCol+1

    # monthly gross income
    dtRegr[:,lastCol] = dtPmts['annual_inc']/12
    print('Monthly gross income: %d' %lastCol)
    lastCol = lastCol+1

    # total outstanding accounts
    dtRegr[:,lastCol] = dtPmts['total_acc']
    print('Total accounts: %d' %lastCol)
    lastCol = lastCol+1

    # revolving utilization
    dtRegr[:,lastCol] = dtPmts['RevolvingLineUtilization']
    print('Revolving utilization: %d' %lastCol)
    lastCol = lastCol+1

    # delinquent accounts in last 2 years
    dtRegr[:,lastCol] = dtPmts['DQ2yrs']
    print('Delinquent accounts in past 2y: %d' %lastCol)
    lastCol = lastCol+1

    # total open credit lines
    dtRegr[:,lastCol] = dtPmts['open_acc']
    print('Total open credit lines: %d' %lastCol)
    lastCol = lastCol + 1

    # home ownership
    dtRegr[np.where(np.logical_or(dtPmts['HomeOwnership'] == 'MORTGAGE',dtPmts['HomeOwnership'] == 'OWN')),lastCol] = 1
    print('Home ownership: %d' %lastCol)
    lastCol = lastCol + 1

    # region
    regions = ['South','Northeast','Pacific','Midwest']
    for i in range(0,len(regions)):
        dtRegr[np.where(dtPmts['region']==regions[i]),lastCol] = 1
        dtCharts['region'].iloc[np.where(dtPmts['region']==regions[i])] = regions[i]
        print('Region: %d' %lastCol)
        lastCol = lastCol+1

    dtCharts['region'].iloc[np.where(pd.isnull(dtCharts['region']))] = 'West'

    return pd.DataFrame(dtRegr,columns=const.regressorFields()), dtCharts

def generate_responses(dtPmts):
# Purpose: Generates dataseries of transitions and transition counts
# 0: current, 1: 1m late, 2: 2m late, 3: 2+m late, 4: default, 5: prepaid entirely

    matCount = np.zeros(shape=(len(const.startStates()),len(const.endStates())),dtype='float')
    dtResp = np.zeros(shape=(dtPmts.shape[0]),dtype='int')

    for i in range(0,len(const.startStates())):
        for j in range(0,len(const.endStates())):
            respIndx = np.where(np.logical_and.reduce((np.array(dtPmts['loan_status'].iloc[1:] == j),
                                                   np.array(dtPmts['loan_status'].iloc[:-1] == i),
                                                   np.array(dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))))
            dtResp[respIndx] = len(const.endStates())*i+j
            matCount[i,j] = len(respIndx[0])
            print('Finished calculating responses for transition state: %s ...' %const.allTransitions()[len(const.endStates())*i+j])

    return pd.Series(data=dtResp,index=dtPmts.index), pd.DataFrame(data=matCount,columns=const.endStates(),index=const.startStates())

def test_train_split(dtPmts,term,test_size):
# Purpose: Splits population into train and test sets for a particular term

    termIndx = np.where(np.in1d(dtPmts['term'],term))
    origIndx = dtPmts['LOAN_ID'].iloc[termIndx]
    origIndx = np.where(np.logical_and(np.in1d(dtPmts['LOAN_ID'],origIndx),dtPmts['MOB']==1))
    X_train, X_test, y_train, y_test = train_test_split(dtPmts.iloc[origIndx], np.ones(shape=(len(origIndx[0]))), test_size=test_size, random_state=0)

    return np.where(np.in1d(dtPmts['LOAN_ID'],dtPmts['LOAN_ID'].iloc[X_train.index])),np.where(np.in1d(dtPmts['LOAN_ID'],dtPmts['LOAN_ID'].iloc[X_test.index]))


def plot_model_output(dtC,superTitle):
# Purpose: Plots model performance vs actuals for several metric buckets

#dtC = dtCharts.iloc[trainingIndx]
#superTitle = 'CtoC'

    #plt.switch_backend('Qt4Agg')
    plt.rcParams.update({'font.size':8})
    f, axArr = plt.subplots(4,4)
    f.suptitle(superTitle)
    terms = [36,60]
    fields = ['MOB','orig_fico','vintage','month','loan_amnt','coupon','purpose','region']

    dtC['y_mod'] = (dtC['y_mod'] == superTitle)
    dtC['y_act'] = (dtC['y_act'] == superTitle)

    for i in range(0,len(terms)):
        for j in range(0,len(fields)):
            print('Charting term: %d and field: %s ... ' %(terms[i],fields[j]))
            mobMod = dtC.groupby(['term',fields[j]])['y_mod'].count() / dtC.groupby(['term'])['y_mod'].count()
            mobAct = dtC.groupby(['term',fields[j]])['y_act'].count() / dtC.groupby(['term'])['y_act'].count()

            if j == 0:
                s0 = axArr[j//2,int(j%2)+2*i].plot(mobAct[terms[i]][0:terms[i]].index,mobAct[terms[i]][0:terms[i]],'-r')
                s1 = axArr[j//2,int(j%2)+2*i].plot(mobMod[terms[i]][0:terms[i]].index,mobMod[terms[i]][0:terms[i]],'-b')

                axArr[j//2,int(j%2)+2*i].legend((s0[0], s1[0]),('Act','Mod'))
                axArr[j//2,int(j%2)+2*i].set_title(fields[j] + ' term=' + str(terms[i]))

            else:
                ind = np.arange(len(mobAct[terms[i]].index))
                width = 0.4

                s0 = axArr[j//2,int(j%2)+2*i].bar(ind,mobAct[terms[i]],width,color='r')
                s1 = axArr[j//2,int(j%2)+2*i].bar(ind+width,mobMod[terms[i]],width,color='b')

                axArr[j//2,int(j%2)+2*i].set_title(fields[j] + ' term=' + str(terms[i]))
                axArr[j//2,int(j%2)+2*i].set_xticks(ind+width)
                axArr[j//2,int(j%2)+2*i].set_xticklabels(mobAct[terms[i]].index)

                # axArr[j//2,int(j%2)+2*i].legend((s0[0], s1[0]),('Act','Mod'))

            #plt.show()
    #f.savefig('/media/koushik/Seagate/Files to Transfer to VM/' + superTitle + '.pdf',dpi=f.dpi)

def analyze_model(model,X_test,y_test,startState):
# Purpose: Provides summary metrics on model performance

#model = modelList[i]
#X_test = dtRegr.iloc[testingIndx]
#y_test = dtResp.iloc[testingIndx]
#startState = 'C'

    # print model accuracy
    print('\nModel overall accuracy is: %.2f for startState: %s ...' \
          % (model.score(X_test,y_test),startState))

    # print OOB error rate
    if (startState == 'GG'):
        print('Model OOB error rate is: %.2f for startState: %s ...' \
             % (1-model.oob_score_,startState))

    # print precision and recall scores
    y_mod = model.predict(X_test)
    print('Model precision and recall scores are: %.4f and %.4f for Pr(%s) ...' \
          % (metrics.precision_score(y_test, y_mod, average='macro'),
             metrics.recall_score(y_test, y_mod, average='macro'), startState))

    # print model vs actual score for each transition
    for c in np.sort(y_test.unique()):
        print('Model : Actual are %.4f : %.4f for Pr(%s) ...' \
              % (float((y_mod == c).sum())/float(y_test.shape[0]),float((y_test == c).sum())/float(y_test.shape[0]),const.allTransitions()[int(c)]))

def generate_transition_models(dtRegr,dtResp,dtCharts,trainIndx,testIndx):
# Purpose: Create random forest models for each transition state

#i = 0
#startState = 'C'
    dtCharts['y_act'] = [const.allTransitions()[int(r)] for r in dtResp]
    modelList = [RandomForestClassifier for i in range(0,len(const.startStates()))]

    for i,startState in enumerate(const.startStates()):

        # select only those rows where the starting state is the same as startState
        # split the relevant data into training and testing sets
        startStateIndx = np.where([colName[0:colName.index('t')] == startState for colName in const.allTransitions()])[0]
        trainingIndx = np.intersect1d(np.where(np.in1d(dtResp,startStateIndx))[0],trainIndx[0])
        testingIndx = np.intersect1d(np.where(np.in1d(dtResp,startStateIndx))[0],testIndx[0])

        # started with GradientBoosting for current loans, but switched to RandomForest model because performance was
        # similar

        #if startState == 'C':
            #model = GradientBoostingClassifier()
        #else:
        model = RandomForestClassifier(n_estimators=const.maxTrees(), max_depth=const.maxDepth(), max_features='auto',
                               bootstrap=True, oob_score=True, random_state=531, min_samples_leaf=const.maxLeaf())

        # create sample with all 6 outcome states for all 4 start states
        sampleIndx = []
        while (dtResp.loc[sampleIndx].value_counts().shape[0] < len(const.endStates())):
            sampleIndx = dtResp.iloc[trainingIndx].sample(const.loanIndxMax(),replace=True).index
            #print(dtResp.loc[sampleIndx].value_counts().shape[0])

        modelList[i] = model.fit(dtRegr.loc[sampleIndx],dtResp.loc[sampleIndx])
        analyze_model(modelList[i],dtRegr.iloc[testingIndx],dtResp.iloc[testingIndx],startState)

        dtCharts['y_mod'].iloc[trainingIndx] = [const.allTransitions()[int(r)] for r in modelList[i].predict(dtRegr.iloc[trainingIndx])]
        dtCharts['y_mod'].iloc[testingIndx] = [const.allTransitions()[int(r)] for r in modelList[i].predict(dtRegr.iloc[testingIndx])]

        # convert dtCharts to floats and generate predicted probs for X
        # gather test data and plot it on bar charts
        for s in startStateIndx:
            #print(s)
            # Turned off plotting for simplicity sake
            plot_model_output(dtCharts.iloc[trainingIndx],const.allTransitions()[int(s)])
            print('Calculated test probs for transition state: %s ... ' %const.allTransitions()[int(s)])

    return modelList, dtCharts


def increment_regressor(X,mob):
# Purpose: Change the input values as we model down the time axis for a loan

    dtModRegr = X.copy(deep=True)
    lastCol = 0

    for i in range(0,10):
        if  (mob >= (i+1)*const.ageKnots()) and (mob < (i+2)*const.ageKnots()):
            dtModRegr.iloc[i] = max(mob - (i+1)*const.ageKnots(),0)
        else:
            dtModRegr.iloc[i] = 0
        lastCol = lastCol+1
        #print(lastCol)

    lastCol = np.where(np.in1d(X.index.values,'age_k0.term36'))[0][0]
    # indicators for term * age linear spline
    for j in range(0,10):
        dtModRegr.iloc[lastCol] = np.multiply(np.absolute(dtModRegr.iloc[11]-1),dtModRegr.iloc[j])
        lastCol = lastCol+1
        #print(lastCol)

    # indicators for term * age linear spline for term=60
    for j in range(0,10):
        dtModRegr.iloc[lastCol] = np.multiply(dtModRegr.iloc[11],dtModRegr.iloc[j])
        lastCol = lastCol+1
        #print(lastCol)

    # loan month
    # determines start month
    lastCol = np.where(np.in1d(X.index.values,'m2'))[0][0]
    if np.where(dtModRegr.iloc[lastCol:lastCol+11]==1)[0].sum()==0:
        month = 1
    else:
        month = np.where(dtModRegr.iloc[lastCol:lastCol+11]==1)[0][0]+2

    # determines new month based on mob
    dtModRegr.iloc[lastCol:lastCol+11] = 0
    if mob + month > 12:
        month = (mob + month) % 12
    else:
        month = mob + month

    # specifies column
    if month > 1:
        dtModRegr.iloc[lastCol+month-2] = 1

    return dtModRegr

def calculate_transition_matrix(X,modelList):
# Purpose: Create the switching probability matrix for a loan payment

    #X = dtRegrOrig.iloc[0]


    # returns cumulative transition prob matrix
    trMat = np.zeros(shape=(len(const.startStates())+2,len(const.endStates())),dtype=float)
    for i in range(0,len(const.startStates())):
        #print(i)
        trMat[i,:] = modelList[i].predict_proba(X.reshape(1,-1))[0]

    trMat[len(const.endStates())-2,len(const.endStates())-2] = 1 # for defs, end in defs
    trMat[len(const.endStates())-1,len(const.endStates())-1] = 1 # for prepays, move to def

    # trMat[0,:] = [.978,.985,.985,.985,.985,1]
    # trMat[1,:] = [.176,.324,.976,.977,.992,1]
    # trMat[2,:] = [.027,.055,.142,.945,.997,1]
    # trMat[3,:] = [.01,.013,.024,.56,998,1]

    #print trMat

    return trMat.cumsum(axis=1)

def simulate_cashflows(coupon,term,loanAmt,xRegr,modelList,stState,nSims):
# Purpose: Simulates CF's for a loan

    np.set_printoptions(precision=4)
    balPaid = np.zeros(shape=(nSims, term), dtype=float)  # cumulative and includes defaults
    prinPrepaid = np.zeros(shape=(nSims, term), dtype=float)  # not cumulative
    prinDef = np.zeros(shape=(nSims, term), dtype=float)  # not cumulative
    intPaid = np.zeros(shape=(nSims, term), dtype=float)
    cfMat = np.zeros(shape=(nSims,term),dtype=float)
    dtState = np.zeros(shape=(nSims,term),dtype=int)

    # Don't do simulation for loans that have paid off or defaulted already
    if (stState < 4):

        if (stState == 0):
            cfMat[:,0] = np.pmt(coupon / 1200, term, loanAmt, 0,0)  # at t=0 all loans are assumed to be current
            balPaid[:,0] = np.ppmt(coupon / 1200, 1, term, loanAmt, 0, 0)
            intPaid[:,0] = np.ipmt(coupon / 1200, 1, term, loanAmt, 0, 0)
            dtState[:,0] = stState

        for j in range(1,term): # j is the time index
            #print(j)
            trMat = calculate_transition_matrix(increment_regressor(xRegr,j),modelList)
            balPaid[:,j] = balPaid[:,j-1]
            stateVar = np.random.uniform(0,1,nSims).reshape(1,-1).transpose() # stateVar are the random uniforms
            dtState[:,j] = np.argmax(np.less(np.tile(stateVar,(1,trMat.shape[1])),trMat[dtState[:,j-1],:]),axis=1)

            if (j == term - 1):
                # terminal condition for current loans or loans prepaid in last time period
                # if current then will be prepaid, if delinquent then will be defaulted
                dtState[np.where(dtState[:,j] == 0),j] = 5
                dtState[np.where(np.logical_and(dtState[:,j] > 0,dtState[:,j] < 4)),j] = 4

            # For prepays, cashFlow = intPaid + prinPaid + rest of loan amt (prinPrepaid)
            prIndx = np.where(np.logical_and(dtState[:,j]==5,dtState[:,j-1]!=5)) # prepay index
            cfMat[prIndx,j] = np.ipmt(coupon/1200,j+1,term,loanAmt,0,0) - loanAmt - balPaid[prIndx,j]
            prinPrepaid[prIndx,j] = -loanAmt - np.ppmt(coupon/1200,j+1,term,loanAmt,0,0) - balPaid[prIndx,j]
            intPaid[prIndx,j] = np.ipmt(coupon/1200,j+1,term,loanAmt,0,0)
            balPaid[prIndx,j] = -loanAmt

            # For defaults, cf = intPaid = prinPrepaid = 0, prinPaid = full loan amount, prinDef = rest of loan amount
            defIndx = np.where(np.logical_and(dtState[:,j]==4,dtState[:,j-1]!=4))
            prinDef[defIndx,j] = -loanAmt - balPaid[defIndx,j]
            balPaid[defIndx, j] = -loanAmt

            # For unequal, states need to add up payments, includes current pmts
            # cf = delta * pmts, prinPaid = delta * ppmt, intPaid = delta * ipmt, prinDef = 0
            lsIndx = np.where(np.logical_and(dtState[:,j]<=dtState[:,j-1],dtState[:,j]<4)) # states are unequal and not prepay or default
            cfMat[lsIndx,j] = np.multiply(np.pmt(coupon/1200,term,loanAmt,0,0),dtState[lsIndx,j-1]-dtState[lsIndx,j]+1)
            balPaid[lsIndx,j] += [np.ppmt(coupon/1200,np.arange(j-dtState[lsIndx[0][l],j-1]+dtState[lsIndx[0][l],j],j+1)+1,term,loanAmt,0,0)[0] \
                                    for l in range(0,lsIndx[0].shape[0])]
            intPaid[lsIndx, j] = [np.ipmt(coupon / 1200, np.arange(j-dtState[lsIndx[0][l],j-1]+dtState[lsIndx[0][l],j],j+1)+1,term,loanAmt,0,0)[0] \
                                    for l in range(0, lsIndx[0].shape[0])]

    return cfMat,prinPrepaid,prinDef,intPaid,balPaid,dtState


def calculate_par_yield(dtSummary,dtOrig,dtRegrOrig,modelList,nSims):
# Purpose: Calculates par credit spread from modelled CF's for a loan (with a flat yield curve)

#dtOrig = dt.iloc[pmtIndx]
#dtRegrOrig = dtR.iloc[pmtIndx]

    loans = np.sort(dtSummary['LoanID'].iloc[np.where(dtSummary['MOB'] == 1)])
    for loan in loans:   # i is the loan index

        i = np.where(dtSummary['LoanID'] == loan)[0][0]
        r = np.where(np.logical_and(dtOrig['LOAN_ID']==loan,dtOrig['MOB']==1))[0][0]

        cfMat = simulate_cashflows(dtSummary['Coupon'].iloc[i],dtSummary['Term'].iloc[i].astype(int),
                                   dtSummary['Amount'].iloc[i],dtRegrOrig.iloc[r],modelList,0,nSims)[0]
        cfMat = np.c_[-dtSummary['Amount'].iloc[i]*np.ones(shape=(cfMat.shape[0])),-1*cfMat]
        summIndx = np.where(dtSummary['LoanID'] == loan)
        dtSummary['ParYield'].iloc[summIndx] = 100*((1+np.irr(cfMat.mean(axis=0)))**12-1)
        print('For loan: %.0f coupon : par yield is %.2f : %.2f ...' % \
              (loan,dtSummary['Coupon'].iloc[summIndx[0][0]],dtSummary['ParYield'].iloc[summIndx[0][0]]))

    return dtSummary

def calculate_actual_yield(dtSummary,dtC,initSwitch):
# Purpose: Calculates the actual yield of a loan
#dtC = dt.iloc[pmtIndx]

    for i in range(0,dtSummary.shape[0]):
        if (initSwitch == True):
            loanIndx = np.where(dtC['LOAN_ID'] == dtSummary['LoanID'].iloc[i])
            loanField = 'loan_amnt'
        else:
            loanIndx = np.where(np.logical_and(dtC['LOAN_ID'] == dtSummary['LoanID'].iloc[i],
                                               dtC['MOB'] > dtSummary['MOB'].iloc[i]))
            loanField = 'PBAL_BEG_PERIOD'

        if (len(loanIndx[0]) > 1):
            cfMat = np.zeros(shape=(len(loanIndx[0]) + 1), dtype=float)
            cfMat[0] = -dtC[loanField].iloc[loanIndx[0][0]]
            cfMat[1:cfMat.shape[0]] = dtC['RECEIVED_AMT'].iloc[loanIndx]
            dtSummary['ActYield'].iloc[i] = 100*((1+np.irr(cfMat))**12-1)
            print('For loan: %.0f @ MOB: %d the actual yield is: %.2f  ...'
                %(dtSummary['LoanID'].iloc[i],dtSummary['MOB'].iloc[i],dtSummary['ActYield'].iloc[i]))

    return dtSummary

def calculate_actual_curves(dtC):
# Purpose: Produces actual CDR and CPR curves for a set of loans

#dtC = dt.iloc[np.where(dt['LOAN_ID'] == dtSummary['LoanID'].iloc[i])]

    dtAct = pd.DataFrame(data=np.zeros(shape=(dtC['term'].iloc[0].astype(int),len(const.curveHeaders())),dtype=float),
                      index=np.arange(1,dtC['term'].iloc[0]+1),columns=const.curveHeaders())
    sumFields = ['PBAL_BEG_PERIOD','scheduled_principal','unscheduled_principal','COAMT','PBAL_END_PERIOD','INT_PAID']
    actFields = ['BegBal','PrinPaid','PrinPrepaid','Defs','EndBal','IntPaid']
    actIndx = np.arange(dtC['MOB'].min().astype(int),dtC['MOB'].max().astype(int)+1)

    dtSumm = dtC.groupby(['MOB'])[sumFields].sum()
    for f in range(0,len(sumFields)):
        dtAct[actFields[f]].loc[actIndx] = dtSumm[sumFields[f]].loc[actIndx].values

    dtAct['NewOrig'].ix[dtC['MOB'].min().astype(int)] = dtAct['BegBal'].ix[dtC['MOB'].min().astype(int)]

    return dtAct

def calculate_performance(dtSummary,dtLast,dtRegrLast,modelList,nSims,initSwitch):
# Purpose: Calculates loan price from par yield

#dtLast = dt.iloc[pmtIndx]
#dtRegrLast = dtR.iloc[pmtIndx]

    #np.set_printoptions(precision=8)
    dtMod = pd.DataFrame(data=np.zeros(shape=(dtLast['term'].max(),len(const.curveHeaders())),dtype=float),
                          index=np.arange(1,dtLast['term'].max()+1),columns=const.curveHeaders())
    dtAct = pd.DataFrame(data=np.zeros(shape=(dtLast['term'].max(),len(const.curveHeaders())),dtype=float),
                          index=np.arange(1,dtLast['term'].max()+1),columns=const.curveHeaders())
    dtCumul = pd.DataFrame(data=np.zeros(shape=(dtLast['term'].max(),len(const.cumulHeaders()))),
                           index=np.arange(1,dtLast['term'].max()+1),columns=const.cumulHeaders())

    for i in range(0,dtSummary.shape[0]):   # i is the loan index

        # if initSwitch then predict all forward CF's, else just ones from current mob + 1 to end
        if (initSwitch == True): # assumes dtLast are only origIndx
            firstMOB = 0
            loanField = 'loan_amnt'
        else:
            firstMOB = dtSummary['MOB'].iloc[i].astype(int)
            loanField = 'PBAL_BEG_PERIOD'

        loanIndx = np.where(np.logical_and(dtLast['LOAN_ID'] == dtSummary['LoanID'].iloc[i],
                                           dtLast['MOB'] == firstMOB + 1))[0]

        if (dtSummary['Term'].iloc[i] - dtSummary['MOB'].iloc[i] > 1) and (len(loanIndx) > 0):
            mobIndx = np.arange(firstMOB+1,dtSummary['Term'].iloc[i].astype(int)+1)
            loanAmt = dtLast[loanField].iloc[loanIndx[0]]
            dtAct = dtAct.add(calculate_actual_curves(dtLast.iloc[np.where(np.logical_and(dtLast['LOAN_ID'] == dtSummary['LoanID'].iloc[i],
                                                                                          dtLast['MOB'] > firstMOB))]))

            cfMat,cfPrepaid,cfDefs,cfInt,cfPaid,dtState = simulate_cashflows(dtSummary['Coupon'].iloc[i],len(mobIndx),
                 loanAmt,dtRegrLast.iloc[loanIndx[0]],modelList,dtSummary['Status'].iloc[i],nSims)

            dtSummary['ParSpread'].iloc[i] = dtSummary['ParYield'].iloc[i] - read_rates()[dtSummary['Term'].iloc[i].astype(int) - 1]
            dtSummary['ModYield'].iloc[i] = 100 * ((np.irr(np.insert(-cfMat.mean(axis=0),0,-loanAmt)) + 1)**12 - 1)
            dtSummary['AvgYield'].iloc[i] = 100 * np.mean([((np.irr(np.insert(-cfMat[row,:],0,-loanAmt)) + 1)**12 - 1) for row in range(0,nSims)])

            parYieldCurve = (.01 * (dtSummary['ParSpread'].iloc[i] + read_rates()[0:len(mobIndx)]) + 1) ** (1.0 / 12)
            discFactor = np.power(parYieldCurve,-np.arange(1,len(mobIndx)+1))

            dtSummary['ModPrice'].iloc[i] = 100 * np.dot(-cfMat.mean(axis=0),discFactor) / loanAmt
            dtSummary['AvgPrice'].iloc[i] = 100 * np.multiply(-cfMat,np.tile(discFactor,(cfMat.shape[0],1))).sum(axis=1).mean() / loanAmt
            dtSummary['ActPrice'].iloc[i] = 100 # temporary until get actual pricing

            dtMod['NewOrig'].loc[firstMOB+1] += loanAmt
            dtMod['PrinPrepaid'].loc[mobIndx] += -cfPrepaid.mean(axis=0) # prepays made positive
            dtMod['Defs'].loc[mobIndx] += -cfDefs.mean(axis=0) # defaults made positive
            dtMod['IntPaid'].loc[mobIndx] += -cfInt.mean(axis=0) # intPaid
            dtMod['PrinPaid'].loc[mobIndx] += np.c_[-cfPaid[:,0],np.diff(-cfPaid,n=1,axis=1)].mean(axis=0) + cfPrepaid.mean(axis=0) + cfDefs.mean(axis=0) # sch prinPaid
            dtMod['EndBal'].loc[mobIndx] += loanAmt + cfPaid.mean(axis=0) # endBal

            if dtMod['EndBal'].iloc[-1] > 1000:
                print('Problem with loan: %d MOB: %d....' %(dtSummary['LoanID'].iloc[i],dtSummary['MOB'].iloc[i]))
                arrs = [cfMat, cfPrepaid, cfDefs, cfInt, cfPaid, dtState]
                names = ['cfMat', 'cfPrepaid', 'cfDefs', 'cfInt', 'cfPaid', 'dtState']
                for k, a in enumerate(arrs):
                    pd.DataFrame(a, columns=mobIndx).to_csv(names[k] + '.csv')

            print('Mod price: avg price is %.2f : %.2f for loan: %d @mob: %d' %(dtSummary['ModPrice'].iloc[i],dtSummary['AvgPrice'].iloc[i],
                                                                                dtSummary['LoanID'].iloc[i],dtSummary['MOB'].iloc[i]))

    dtMod['BegBal'] = dtMod[['EndBal','PrinPaid','PrinPrepaid','Defs']].sum(axis=1)  # next begBal is last endBal
    dtMod['CPR'] = (dtMod['PrinPrepaid']/dtMod['BegBal']).fillna(0)
    dtMod['CDR'] = (dtMod['Defs']/dtMod['BegBal']).fillna(0)
    dtAct['CPR'] = (dtAct['PrinPrepaid']/dtAct['BegBal']).fillna(0)
    dtAct['CDR'] = (dtAct['Defs']/dtAct['BegBal']).fillna(0)
    dtComp = pd.merge(dtMod,dtAct,how='outer',left_index=True,right_index=True,suffixes=('_m','_a'))
    dtCumul[['CPR_mod','CDR_mod']] = 100 * dtMod[['PrinPrepaid','Defs']].cumsum(axis=0) / dtMod['NewOrig'].sum(axis=0)
    dtCumul[['CPR_act','CDR_act']] = 100 * dtAct[['PrinPrepaid','Defs']].cumsum(axis=0) / dtAct['NewOrig'].sum(axis=0)

    return dtSummary,dtComp,dtCumul

def calculate_summary(dt,dtR,numLoans,term,modelList,nSims,initSwitch):
# Purpose: Outputs modelled vs actual CPR and CDR curves as well as prices and par yields

# numLoans=1
# term=60
# nSims=100
# dt = dtPmts.iloc[testIndx]
# dtR = dtRegr.iloc[testIndx]
# initSwitch = False

    sampleLoans = dt['LOAN_ID'].iloc[np.where(np.logical_and(dt['MOB']==1,dt['term']==term))].sample(numLoans).values
    pmtIndx = np.where(np.logical_and(np.in1d(dt['LOAN_ID'],sampleLoans),dt['MOB']<=dt['term']))
    origIndx = np.where(np.logical_and(np.in1d(dt['LOAN_ID'],sampleLoans),dt['MOB']==1))

    if (initSwitch == True):
        summIndx = origIndx
    else:
        summIndx = pmtIndx

    dtSummary = pd.DataFrame(np.zeros(shape=(len(summIndx[0]), len(const.priceHeaders())), dtype=float),
                            index=dt.iloc[summIndx].index, columns=const.priceHeaders())
    dtSummary[['LoanID','MOB','Term','Status','Amount','Coupon']] = dt[['LOAN_ID','MOB','term','loan_status','loan_amnt','coupon']].iloc[summIndx].values

    dtSummary = calculate_par_yield(dtSummary,dt.iloc[pmtIndx],dtR.iloc[pmtIndx],modelList,nSims)
    dtSummary = calculate_actual_yield(dtSummary,dt.iloc[pmtIndx],initSwitch)
    dtSummary,dtComp,dtCumul = calculate_performance(dtSummary,dt.iloc[pmtIndx],dtR.iloc[pmtIndx],modelList,nSims,initSwitch)

    dtSummary.to_csv('Curves/summ' + du.datetime.today().strftime('%Y%m%d') + '.csv')
    dtComp.to_csv('Curves/comp' + du.datetime.today().strftime('%Y%m%d') + '.csv')
    dtCumul.to_csv('Curves/cum' + du.datetime.today().strftime('%Y%m%d') + '.csv')

    return dtSummary,dtComp,dtCumul



def main(argv = sys.argv):

    quandl.ApiConfig.api_key = const.quandlKey()
    dtPmts = read_pmt_data('LC Data/PMTHIST_ALL_20170215.csv')
    dtPmts = clean_pmt_data(dtPmts)
    dtPmts = check_pmt_data(dtPmts)
    dtPmts = insert_status(dtPmts)
    dtPmts = read_stats_data(dtPmts)
    dtPmts = pre_processing(dtPmts)

    dtRegr, dtCharts = generate_regressors(dtPmts)
    dtResp, matCount = generate_responses(dtPmts)
    trainIndx, testIndx = test_train_split(dtPmts,[36,60],.33)
    modelList, dtCharts = generate_transition_models(dtRegr,dtResp,dtCharts,trainIndx,testIndx)

    dtSummary,compCurves,cumCurves = calculate_summary(dtPmts.iloc[testIndx],dtRegr.iloc[testIndx],10,60,modelList,100,True)
    dtSummary,compCurves,cumCurves = calculate_summary(dtPmts.iloc[testIndx],dtRegr.iloc[testIndx],10,60,modelList,100,False)

if __name__ == "__main__":
    sys.exit(main())