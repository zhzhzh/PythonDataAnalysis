
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[1]:

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import re
from pandas import DataFrame


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[2]:

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[3]:

len(states)


# In[4]:

def clean_state(ss: str):
    state = re.sub(r'\[.*', '',ss).strip()
    # print('{}=>{}'.format(ss, state))
    return state


# In[5]:

def clean_region(reg: str):
    region = re.sub(r'\(.*', '', reg).strip()
    # print('{}==>{}'.format(reg, region))
    return region


# In[6]:

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    
    with open('university_towns.txt', 'r') as in_file:
        town_list = []
        state = ''
        for line in in_file.readlines():
            line = line.strip()
            # print(line)
            if '[edit]' in line:
                state = clean_state(line)
                continue
            else:
                region = clean_region(line)
                town_list.append([state, region])

        df = DataFrame(town_list, columns=["State", "RegionName"])
    return df


# In[7]:

uni_towns = get_list_of_university_towns()


# In[9]:

def check_ind(x):
    if x > 0:
        return 1
    else:
        return -1


# In[10]:

def read_gdp():
    dd = pd.read_excel(
        'gdplev.xls',
        skiprows=219,
        parse_cols='E:G',
        names=['quarter', 'dollars', 'dollars_2009']
    )
    dd['tmp'] = dd['dollars_2009'].shift(1)
    dd['dollar_diff'] = dd['dollars_2009'] - dd['tmp']
    dd['dollar_ind'] = dd['dollar_diff'].apply(check_ind)
    dd['dollar_rolling'] = dd['dollar_ind'].rolling(2).sum()
    return dd


# In[11]:

dd = read_gdp()


# In[12]:

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    dd = read_gdp()
    # quarter = dd[dd['dollar_rolling'] == -2]['quarter'].values[0]
    dd2 = dd[dd['dollar_rolling'] == -2]
    quarter = dd.iloc[dd2.index[0] - 1]['quarter']
    return quarter


# In[13]:

get_recession_start()


# In[14]:

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    dd = read_gdp()
    df2 = dd[dd['dollar_rolling'] == 2]
    quarter = df2[df2['quarter'] > get_recession_start()]['quarter'].values[0]    
    return quarter


# In[15]:

get_recession_end()


# In[16]:

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    dd = read_gdp()
    dd2 = dd[dd['quarter'] >= get_recession_start()]
    dd2 = dd2[dd['quarter'] <= get_recession_end()]
    quarter = dd.iloc[dd2['dollars_2009'].idxmin()]['quarter']
    return quarter


# In[17]:

get_recession_bottom()


# In[18]:

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    
    df3 = pd.read_csv('City_Zhvi_AllHomes.csv')
    df3['State'] = df3['State'].map(states)
    # df3 = df3.head()
    df_tmp = df3.T[6:]
    df_tmp.index = pd.to_datetime(df_tmp.index).to_period(freq='Q')
    df_tmp = df_tmp[df_tmp.index >= '2000Q1']
    df_tmp = pd.DataFrame(df_tmp, dtype=np.float)
    df_tmp2 = df_tmp.groupby(df_tmp.index).mean()
    df_tmp2.index = df_tmp2.index.to_series().astype(str).str.lower()
    df4 = df_tmp2.T
    df5 = df3[['State', 'RegionName']]
    df6 = pd.concat([df5, df4], axis=1)
    return df6.set_index(["State","RegionName"])


# In[82]:

convert_housing_data_to_quarters().head()


# In[26]:

def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    
    start = get_recession_start()
    end = get_recession_bottom()
    data = convert_housing_data_to_quarters().copy()

    data = data.loc[:, start:end]
    data = data.reset_index()
    data['price_ratio'] = data[start]/data[end]

    uni_towns = get_list_of_university_towns()
    uni_towns['is_uni'] = 1
    new_data = pd.merge(data, uni_towns, how='left', on=['State', 'RegionName'])
    new_data['is_uni'] = new_data['is_uni'].fillna(0)

    df_non_uni = new_data[new_data['is_uni'] == 0]['price_ratio']
    df_uni = new_data[new_data['is_uni'] == 1]['price_ratio']
    non_uni_mean = df_non_uni.mean()
    uni_mean = df_uni.mean()

    statistic, p = ttest_ind(df_non_uni, df_uni, nan_policy='omit')
    ttest_ind(df_uni, df_non_uni, nan_policy='omit')

    difference = None
    better = None

    if p < 0.01:
        difference = True
    else:
        difference = False


    if uni_mean < non_uni_mean:
        better = "university town"
    else:
        better = "non-university town"

    
    result = (difference, p, better)
    
    
    return result


# In[ ]:



