# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:49:20 2024

@author: kokol
"""

from typing import List
import os
from os.path import abspath, join
import yfinance as yf
import pandas as pd

current_directory = os.getcwd()
print(abspath(current_directory))

def get_data_init(ticker: str = 'MSFT', period: str = 'max', filter_years: List[int] = None) -> pd.DataFrame:

    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    data = hist.reset_index()

    data['day_of_week'] = data['Date'].dt.dayofweek
    data['day_of_month'] = data['Date'].dt.day
    data['month_of_year'] = data['Date'].dt.month
    data['quarter_of_year'] = data['Date'].dt.quarter
    data['year'] = data['Date'].dt.year

    if filter_years is not None:
        data = data.loc[data['year'].isin(filter_years)]

    return data

df = get_data_init()
df.to_csv(abspath(join(current_directory, 'MSFT_data.csv')))

