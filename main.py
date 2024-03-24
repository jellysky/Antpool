# __author__ = 'koushik'

import requests
import time
import hmac , hashlib
import pandas as pd
import numpy as np
import datetime as dtt
import nasdaqdatalink as ndl
import dateutils as du
import matplotlib.pyplot as plt

class const():
    @staticmethod
    def coinType():
        return 'BTC'
    @staticmethod
    def signID():
        return 'serention'
    @staticmethod
    def signKey():
        return 'd41e105f4a5548dc9796dcf84f514dc4'
    @staticmethod
    def signSecret():
        return 'c473a0e110794f5f88578ea4a2d18783'
    @staticmethod
    def userID():
        return 'jellysky'
    @staticmethod
    def userWorkerID():
        return '1'
    @staticmethod
    def fixedExpenses():
        return 35/30
    @staticmethod
    def globalRewards():
        return 927
    @staticmethod
    def halvingDate():
        return dtt.datetime(2024,5,1)
    def dailyHRGrowth():
        return 2**(1/(365*2))-1
    @staticmethod
    def minerUptime():
        return .95
    @staticmethod
    def initialMinerCost():
        return 320
    @staticmethod
    def forecastWindow():
        return 365
    @staticmethod
    def startDate():
        return dtt.datetime(2023,11,2)
    @staticmethod
    def hedgePrice():
        return 37000
    @staticmethod
    def hedgeUnits():
        return .0122

def pickle_save(dt,saveStr):
# Purpose: Stores pandas dataframe as a pickled .p file

    pk.dump(dt, open('Pickled/'+saveStr, 'wb'))
    print('Pickled file ...')

def pickle_load(loadStr):
# Purpose: Loads object from a pickled .p file

    dt = pd.DataFrame()

    for l in loadStr:
        print('Loaded pickled file from: %s...' %l)
        dt = pd.concat([dt,pk.load(open('Pickled/'+l, 'rb'))],axis=0)

    return dt

def pull_nasdaqapi_hr(tickers,columns):

    ndl.read_key('.nasdaq/data_link_apikey')
    dt = ndl.get(tickers)
    dt.columns = columns
    dt.index.name = 'Date'
    dt.drop(dt.index[np.where(dt.index < const.startDate())],axis=0,inplace=True)

    return dt

def urlDict(pullStr, apiSign, hashInput):
    urlDict = {
        'poolStats': ['https://antpool.com/api/poolStats.htm', {'key': const.signKey(), 'nonce': apiSign[1], 'signature': apiSign[0], 'coin': const.coinType()}],
        'accountBalance': ['https://antpool.com/api/account.htm', {'key': const.signKey(), 'nonce': apiSign[1], 'signature': apiSign[0], 'coin': const.coinType()}],
        'userHashrate': ['https://antpool.com/api/hashrate.htm', {'key': const.signKey(), 'nonce': apiSign[1], 'signature': apiSign[0],'coin': const.coinType()}],
        'workerHashrate': ['https://antpool.com/api/workers.htm', {'key': const.signKey(), 'nonce': apiSign[1], 'signature': apiSign[0],'coin': const.coinType()}],
        'accountPayments': ['https://antpool.com/api/paymentHistory.htm', {'key': const.signKey(), 'nonce': apiSign[1], 'signature': apiSign[0],'coin': const.coinType()}],
        'queryHashrate': ['https://antpool.com/api/userHashrateChart.htm', {'key': const.signKey(), 'nonce': apiSign[1], 'signature': apiSign[0], 'coin': const.coinType(),'userId': const.signID(), 'type': 3}],
        'workerList': ['https://antpool.com/api/userWorkerList.htm', {'key': const.signKey(), 'nonce': apiSign[1], 'signature': apiSign[0], 'coin': const.coinType(),'userId': const.signID(), 'workerStatus': 0}],
        'queryPayments': ['https://antpool.com/api/paymentHistoryV2.htm', {'key': const.signKey(), 'nonce': apiSign[1], 'signature': apiSign[0], 'coin': const.coinType(), 'type': 'recv', 'pageSize': 50}],
        'minerCalc': ['https://antpool.com/api/coinCalculator.htm', {'key': const.signKey(), 'nonce': apiSign[1], 'signature': apiSign[0], 'coinType': 'BTC', 'hashInput': hashInput}]
    }
    return urlDict[pullStr]

def get_signature():

    nonce = int(time.time()*1000)
    msgs = const.signID() + const.signKey() + str(nonce)
    ret = []
    ret.append(hmac.new(const.signSecret().encode(encoding="utf-8"), msg=msgs.encode(encoding="utf-8"), digestmod=hashlib.sha256).hexdigest().upper())
    ret.append(nonce)

    return ret

def pull_antpoolapi_hr():

#pullStr = 'minerCalc'
    postData = urlDict('queryHashrate',get_signature(),0)
    request = requests.post(postData[0], postData[1]).text
    dt = pd.DataFrame.from_dict(pd.read_json(request)['data'][0]['poolSpeedBeanList'])
    dt['date'] = dt['date'] / 1000
    dt.index = pd.to_datetime(dt['date'].apply(lambda x: dtt.datetime.utcfromtimestamp(x).strftime('%m-%d-%Y')),infer_datetime_format=True)
    dt.drop(['date'],axis=1,inplace=True)
    dt.drop(dt.index[np.where(dt.index < const.startDate())], axis=0, inplace=True)

    return dt.sort_index()

def pull_antpoolapi_pmts():

    postData = urlDict('queryPayments',get_signature(),0)
    request = requests.post(postData[0], postData[1]).text
    dt = pd.DataFrame.from_dict(pd.read_json(request)['data']['rows'])
    dt.index = pd.to_datetime(dt['timestamp'], infer_datetime_format=True)
    dt['hashrate_num'] = pd.to_numeric(dt['hashrate'].str[:4])
    dt['fppsBlockAmount'] = pd.to_numeric(dt['fppsBlockAmount'])
    dt['fppsFeeAmount'] = pd.to_numeric(dt['fppsFeeAmount'])

    return dt[['fppsBlockAmount','fppsFeeAmount']].sort_index()

def calculate_usdrewards():

    dt = pull_antpoolapi_pmts()
    dtBTCPrice = pull_nasdaqapi_hr(['BCHAIN/MKPRU'], ['BTCPrice'])
    dt = dt.merge(dtBTCPrice, how='outer', left_index=True, right_index=True)
    dt['fppsBlockAmountUSD'] = dt['fppsBlockAmount'] * dt['BTCPrice']
    dt['fppsFeeAmountUSD'] = dt['fppsFeeAmount'] * dt['BTCPrice']

    return dt[['BTCPrice','fppsBlockAmount','fppsFeeAmount','fppsBlockAmountUSD','fppsFeeAmountUSD']].sort_index()

def pull_antpoolapi_minercalc(hashInput):

    postData = urlDict('minerCalc',get_signature(),hashInput)
    request = requests.post(postData[0], postData[1]).text

    return request

def project_network_hr(startDate,endDate,lastHR):

    dt = pd.DataFrame(data=0,index=pd.date_range(startDate,endDate),columns=['HR'])
    dt['HR'] = lastHR * np.power((1 + const.dailyHRGrowth()), np.arange(1, dt.shape[0] + 1, 1))
    return dt

def generate_chart_data():

    dtOut = pd.DataFrame(data=0,
                         index=pd.date_range(const.startDate(),const.startDate() + du.relativedelta(days=const.forecastWindow())),
                         columns=['BTCPriceActual','BTCPriceProjected','BTCPrice','GlobalBTCRewards',
                                  'MinerHRActual', 'MinerHRProjected', 'NetworkHRActual', 'NetworkHRProjected',
                                  'MinerBTCRewardsActual', 'MinerBTCRewardsProjected', 'CumMinerBTCRewardsActual', 'CumMinerBTCRewardsProjected',
                                  'MinerRevenueActual', 'MinerRevenueProjected', 'MinerExpensesActual', 'MinerExpensesProjected','MinerPNLActual','MinerPNLProjected',
                                  'MinerPNL','CumMinerPNL','HedgePNL','CumHedgePNL','NetPNL','CumNetPNL','PayoffTime'],
                         dtype=float)
    dtOut.index.name = 'Date'

    # for calculation of hashrate related metrics
    dtActMn = pull_antpoolapi_hr()
    dtActNw = pull_nasdaqapi_hr(['BCHAIN/HRATE'],['BTCHR'])
    dtProjNw = project_network_hr(dtActNw.index[-1] + du.relativedelta(days=+1),
                                  const.startDate() + du.relativedelta(days=+365),
                                  dtActNw['BTCHR'].iloc[-1])
    dtPmts = calculate_usdrewards()

    # these are values utilized in several calculations
    dtOut.loc[dtPmts.index,'BTCPriceActual'] = dtPmts['BTCPrice'].values
    zeroIndx = dtOut.index[np.where(dtOut['BTCPriceActual'] == 0)]
    dtOut.loc[zeroIndx,'BTCPriceProjected'] = np.random.normal(0,10000,len(zeroIndx)) + const.hedgePrice()
    dtOut['GlobalBTCRewards'] = const.globalRewards()
    dtOut.loc[dtOut.index >= const.halvingDate(), 'GlobalBTCRewards'] = 0.5 * const.globalRewards()
    dtOut['BTCPrice'] = dtOut[['BTCPriceActual','BTCPriceProjected']].sum(axis=1)

    # for calculation of btc denominated rewards metrics
    dtOut.loc[dtActMn.index,'MinerHRActual'] = np.concatenate(dtActMn.values, axis=0)
    dtOut.loc[dtOut.index[np.where(dtOut.index > dtActMn.index[-1])],'MinerHRProjected'] = dtActMn['speed'].mean()
    dtOut.loc[dtActNw.index,'NetworkHRActual'] = dtActNw['BTCHR']
    dtOut.loc[dtProjNw.index,'NetworkHRProjected'] = dtProjNw['HR']

    #presentIndx = dtPmts.index
    #futureIndx = dtOut.index[np.where(dtOut['NetworkHRProjected'] > 0)]

    dtOut.loc[dtPmts.index,'MinerBTCRewardsActual'] = dtPmts[['fppsBlockAmount','fppsFeeAmount']].sum(axis=1,skipna=True).values
    dtOut.loc[zeroIndx,'MinerBTCRewardsProjected'] = const.minerUptime() * dtOut['GlobalBTCRewards'] * dtOut['MinerHRProjected'] / dtOut['NetworkHRProjected']
    #dtOut.fillna(0,inplace=True)
    #dtOut.replace([np.inf, -np.inf], 0, inplace=True)
    dtOut[['CumMinerBTCRewardsActual','CumMinerBTCRewardsProjected']] = dtOut[['MinerBTCRewardsActual','MinerBTCRewardsProjected']].cumsum(axis=0)
    #dtOut.drop(['GlobalRewards'],axis=1,inplace=True)

    # for calculation of usd denominated rewards metrics
    dtOut.loc[dtPmts.index,'MinerRevenueActual'] = dtPmts[['fppsBlockAmountUSD','fppsFeeAmountUSD']].sum(axis=1,skipna=True).values
    dtOut['MinerRevenueProjected'] = dtOut['BTCPriceProjected'] * dtOut['MinerBTCRewardsProjected']
    dtOut.loc[dtOut.index[np.where(dtOut['BTCPriceActual'] > 0)],'MinerExpensesActual'] = const.fixedExpenses()
    dtOut.loc[zeroIndx,'MinerExpensesProjected'] = const.fixedExpenses()
    dtOut['MinerPNLActual'] = dtOut['MinerRevenueActual'] - dtOut['MinerExpensesActual']
    dtOut['MinerPNLProjected'] = dtOut['MinerRevenueProjected'] - dtOut['MinerExpensesProjected']

    # for calculation of daily pnl's
    dtOut['MinerPNL'] = dtOut[['MinerPNLActual','MinerPNLProjected']].sum(axis=1,skipna=True)
    dtOut['CumMinerPNL'] = dtOut['MinerPNL'].cumsum(axis=0)
    dtOut['CumHedgePNL'] = (const.hedgePrice() - dtOut['BTCPrice']) * const.hedgeUnits()
    dtOut['HedgePNL'] = dtOut['CumHedgePNL'].diff()
    dtOut.at[dtOut.index[0],'HedgePNL'] = (const.hedgePrice() - dtOut['BTCPrice'].iloc[0]) * const.hedgeUnits()
    dtOut['NetPNL'] = dtOut[['MinerPNL','HedgePNL']].sum(axis=1,skipna=True)
    dtOut['CumNetPNL'] = dtOut['MinerPNL'].cumsum(axis=0)

    poIndx = np.argmax(dtOut['CumNetPNL'].values > const.initialMinerCost())
    dtOut.loc[dtOut.index[poIndx],'PayoffTime'] = 1

    dtOut.to_csv('dtOut.csv')

    return dtOut

    #def plot_charts():

    superTitle = 'Hedged Mining Performance'

    plt.rcParams.update({'font.size':8})
    f, axArr = plt.subplots(2,2)
    f.suptitle(superTitle)

    print('Charting hashrate over time: ')
    s0 = axArr[0, 0].plot(dtOut.index, dtOut['MinerHRActual'], '-r')
    s1 = axArr[0, 0].plot(dtOut.index, dtOut['MinerHRProjected'], '-g')
    s2 = axArr[0, 0].plot(dtOut.index, dtOut['NetworkHRActual'], '-b')
    s3 = axArr[0, 0].plot(dtOut.index, dtOut['NetworkHRProjected'], '-p')

    axArr[j // 2, int(j % 2) + 2 * i].legend((s0[0], s1[0]), ('Act', 'Mod'))
    axArr[j // 2, int(j % 2) + 2 * i].set_title(fields[j] + ' term=' + str(terms[i]))


    f.show()

    s1 = axArr[j // 2, int(j % 2) + 2 * i].plot(mobMod[terms[i]][0:terms[i]].index, mobMod[terms[i]][0:terms[i]], '-b')







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

def main():

dtOut = generate_chart_data()
#datetime.datetime.strptime(dt['date'].iloc[0],"%d%m%Y")
#print(datetime.datetime.utcfromtimestamp(tt/1000).strftime('%Y-%m-%d %H:%M:%S'))


main()
