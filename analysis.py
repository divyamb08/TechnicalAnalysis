import ccxt
import pandas as pd
# pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

exchange = ccxt.binance()

bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d')
df = pd.DataFrame(bars, columns=['timestamp','open','high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True, drop =True)
# print(df)
def tr(df):
    df['previous_close'] = df['close'].shift(1)
    df['%-change'] = df['close'].pct_change()
    df["high-low"] = abs(df['high'] - df['low'])
    df['high-pc'] = abs(df['high'] - df['previous_close'])
    df['low-pc'] = abs(df['low'] - df['previous_close'])
    TR = df[['high-low','high-pc', 'low-pc']].max(axis=1)
    return TR

def atr(df, period):
    df['TR'] = tr(df)
    atr = df['TR'].rolling(period).mean()
    return atr

def analysis(df, period, multiplier):
    print("Analyzing")
    df['ATR'] = atr(df, period = period)
    df['UpperBand'] = ((df['high'] + df['low'])/2 + (multiplier * df['TR']))
    df['LowerBand'] = ((df['high'] + df['low'])/2 - (multiplier * df['TR']))
    df['in_uptrend'] = True
    for current in range(1, len(df.index)):
        previous = current-1
        if df['close'][current]> df['UpperBand'][previous]:
            df['in_uptrend'][current] = True
        elif df['close'][current]<df['LowerBand'][previous]:
            df['in_uptrend'][current] = False
        else:
            df['in_uptrend'][current] = df['in_uptrend'][previous]
            if df['in_uptrend'][current] and df['LowerBand'][current]<df['LowerBand'][previous]:
                df['LowerBand'][current] = df['LowerBand'][previous]
            
            if not df['in_uptrend'][current] and df['UpperBand'][current]>df['UpperBand'][previous]:
                df['UpperBand'][current] = df['UpperBand'][previous]
             
    df = df.drop((['high-low', 'high-pc', 'low-pc', 'TR', 'previous_close']), axis=1)
    return df

def get_fib_retracement(df):
    max_price = df['close'].max()
    min_price = df['close'].min()
    diff = max_price - min_price
    
    first_level = max_price-diff*0.236
    second_level = max_price-diff*0.382
    third_level = max_price-diff*0.5
    fourth_level = max_price-diff*0.618
    plt.figure(figsize=(12.33,4.5))
    
    plt.plot(df['high'],label='high')
    
    
    pivots = []
    dates = []
    counter=0
    lastpivot=0
    Range = [0] * 10
    dateRange = [0] * 10
    for i in df.index:
        current_max = max(Range, default=0)
        value = round(df['high'][i], 2)
        Range = Range[1:9]
        Range.append(value)
        
        dateRange = dateRange[1:9]
        dateRange.append(i)
        
        if current_max == max(Range, default=0):
            counter += 1
        else:
            counter=0
        if counter == 5:
            lastpivot = current_max
            dateloc = Range.index(lastpivot)
            lastDate = dateRange[dateloc]
            pivots.append(lastpivot)
            dates.append(lastDate)
    
    timeD = timedelta(days=30)
    
    for i in range(len(pivots)) :
        print(str(pivots[i]) + "\t" + str(dates[i]))
        # plt.figure(figsize=(12.33,4.5))
        plt.title("Fibonacci retracement plot")
        plt.plot_date([dates[i], dates[i]+timeD], [pivots[i], pivots[i]], linestyle = "-", color='red',linewidth=2, marker=",")
        # plt.plot(df.index, df['close'])
        plt.axhline(max_price, linestyle='--', alpha=0.5,color='brown')
        plt.axhline(first_level, linestyle='--', alpha=0.5,color='orange')
        plt.axhline(second_level, linestyle='--', alpha=0.5,color='yellow')
        plt.axhline(third_level, linestyle='--', alpha=0.5,color='green')
        plt.axhline(fourth_level, linestyle='--', alpha=0.5,color='blue')
        plt.axhline(min_price, linestyle='--', alpha=0.5,color='purple')
        
    plt.show()
    # new_df = df
    # new_df.set_index('timestamp')
    # plt.figure(figsize=(12.33,4.5))
    # plt.title("Fibonacci retracement plot")
    # plt.plot(new_df.index, new_df['close'])
    # plt.axhline(max_price, linestyle='--', alpha=0.5,color='red')
    # plt.axhline(first_level, linestyle='--', alpha=0.5,color='orange')
    # plt.axhline(second_level, linestyle='--', alpha=0.5,color='yellow')
    # plt.axhline(third_level, linestyle='--', alpha=0.5,color='green')
    # plt.axhline(fourth_level, linestyle='--', alpha=0.5,color='blue')
    # plt.axhline(min_price, linestyle='--', alpha=0.5,color='purple')
    # plt.xlabel("Date")
    # plt.ylabel("Price")
    # plt.show()
    
    
def get_maxima_minima(df):
    n=5
    df['minima'] = df.iloc[argrelextrema(df.close.values, np.less_equal,
                    order=n)[0]]['close']
    df['maxima'] = df.iloc[argrelextrema(df.close.values, np.greater_equal,
                    order=n)[0]]['close']
    # plt.scatter(df.index, df['minima'], c='r')
    # plt.scatter(df.index, df['maxima'], c='g')
    # plt.plot(df.index, df['close'])
    # plt.show()
    return df


def calc_volatility(df):
    # df = tr(df)
    df['Log Returns'] = np.log(df['close']/df['close'].shift())
    volatility = df['Log Returns'].std()*252**.5
    str_vol = str(round(volatility,3)*100)
    print("The volatility is {}".format(str_vol))

def moving_avg(df, x):
    df['MA {}'.format(x)] = df['close'].rolling(x).mean()
    return df

def exp_moving_avg(df, x):
    df['EMA {}'.format(x)] = df['close'].ewm(span=10, adjust=False).mean()
    return df

def get_volume(df):
    df['Average DV'] = pd.rolling_mean(df['volume'], window=5)
    

# calc_volatility(df)
# moving_avg(df, 10)
# exp_moving_avg(df, 10)

# print(analysis(df, 15, 3))
# print(get_maxima_minima(df))
get_fib_retracement(df)
