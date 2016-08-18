#-*- coding: utf-8 -*-
import datetime
import requests, json
import BeautifulSoup

import pandas as pd
import pandas.io.data as web

import matplotlib.pyplot as plt
from common import *



def download_stock_data(file_name,company_code,year1,month1,date1,year2,month2,date2):
	start = datetime.datetime(year1, month1, date1)
	end = datetime.datetime(year2, month2, date2)
	df = web.DataReader("%s.KS" % (company_code), "yahoo", start, end)

	df.to_pickle(file_name)

	return df


def read_stock_code_from_xls(file_name):
	df_code = pd.read_excel(file_name)

	return df_code


def download_whole_stock_data(market_type,year1,month1,date1,year2,month2,date2):
	df_code = read_stock_code_from_xls('stock_code.xls')

	for index in range(df_code.shape[0]):
		stock_code = df_code.loc[index,df_code.columns[0]]
		name = df_code.loc[index,df_code.columns[1]]
		market = df_code.loc[index,df_code.columns[2]]

		if market_type.upper()=='KOSPI':
			print "... downloading %s of %s : code=%s, name=%s" % (index+1, df_code.shape[0], stock_code,name)
			try:
				full_file_name = get_data_file_path(stock_code)
				download_stock_data('%s.data'%(full_file_name),stock_code,year1,month1,date1,year2,month2,date2)
			except:
				print "... Error!!! while downloading stock data"


#download_stock_data('samsung.data','005930',2010,1,1,2016,1,30)
#download_stock_data('lg.data','066570',2010,1,1,2016,1,30)

#read_stock_code_from_xls('stock_code.xls')
download_whole_stock_data('kospi',2010,1,1,2016,2,1)