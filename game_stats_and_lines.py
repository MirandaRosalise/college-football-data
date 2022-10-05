# Databricks notebook source
import requests
import json
import pandas as pd

# COMMAND ----------

#api_check

api_key='redacted'
test_payload={'year': '2019'}
test_headers = {'Authorization': 'Bearer ' + api_key}
test_request=requests.get('https://api.collegefootballdata.com/games', 
                          params=test_payload, 
                          headers=test_headers)

test_data = test_request.content
df_test = pd.read_json(test_data)
df_test.head()

# COMMAND ----------

url='https://api.collegefootballdata.com/games'
api_key='redacted'
test_headers = {'Authorization': 'Bearer ' + api_key}

years = [str(n) for n in range(2000, 2022)]

df_games = pd.concat(
    [
        pd.read_json(
            requests.get(url, 
                         params={'year': n},
                         headers=test_headers
                        ).content
        ) for n in years
    ]
)

df_games.head()

# COMMAND ----------

url='https://api.collegefootballdata.com/lines'

fetch_lines = [
    requests.get(url, 
        params={'year': n},
        headers=test_headers
        )
    for n in years
]
fetch_lines

# COMMAND ----------

df_lines = pd.concat([pd.read_json(n.content) for n in fetch_lines])

# COMMAND ----------

df_lines

# COMMAND ----------

lines_fixed = (
    pd.merge(
        left=df_lines, 
        right=pd.concat(
            [pd.DataFrame(x) for x in df_lines.lines], 
                keys = df_lines.id
            ).reset_index(),
        left_on='id',
        right_on='id'
        )
    .drop('level_1', axis=1)
    .set_index(['id', 'provider'])
)
lines_fixed.head()

# COMMAND ----------

import math
import random

lines_fixed['actual_spread'] = (
    lines_fixed.awayScore-lines_fixed.homeScore
)

lines_fixed['B'] = lines_fixed.actual_spread-lines_fixed.spread.astype(float)
lines_fixed.dropna(subset=['B'], inplace=True)
lines_fixed.B = lines_fixed.B.apply(lambda x: math.floor(x+random.random()))
lines_fixed.B.head()

# COMMAND ----------

from scipy.stats.mstats import winsorize

lines_fixed['BW'] = winsorize(lines_fixed.B, limits=[0.05,0.05])
lines_fixed.BW.describe()

# COMMAND ----------

from scipy.stats import shapiro as shap
import random

lines_fixed
print(shap(lines_fixed.B.dropna().sample(5000)))
print(shap(lines_fixed.BW.dropna().sample(5000)))
lines_fixed.B.hist(bins=100)

# COMMAND ----------

lines_fixed

# COMMAND ----------

#fetch every game's advanced stats

url='https://api.collegefootballdata.com/games/teams'
test_payload={'year': '2019', 'gameId': '401112224'}
test_headers = {'Authorization': 'Bearer ' + api_key}
test_request = requests.get(url=url,
                           params=test_payload,
                           headers=test_headers)
test_request

# COMMAND ----------

import yaml
import json

print(yaml.dump(json.loads(test_request.content.decode())))

# COMMAND ----------

params_list = list(
    zip(lines_fixed.groupby(level=0).first().season,
        lines_fixed.index.levels[0])
)

params_list = [p for p in params_list if p[0]==2021]
responses = []
url='https://api.collegefootballdata.com/games/teams'
headers={'Authorization': 'Bearer ' + api_key}

for p in params_list:
    payload = {'year': p[0],
               'gameId': p[1]}
    responses.append(requests.get(url=url,
                                  params=payload,
                                  headers=headers))

responses

# COMMAND ----------

import os

path = ''

for i in range(0, len(responses)):
    filename = str(params_list[i][1]) + '.json'
    filename = os.path.join(path, filename)
    with open(filename, 'w') as f:
        f.write(responses[i].content.decode())

# COMMAND ----------

#pipe to process response content into a usable dataframe

df_stats = pd.concat([pd.read_json(r.content) for r in responses], ignore_index=True)
df_stats = pd.concat([pd.DataFrame(x) for x in df_stats.teams],
                      keys=df_stats.id)
df_stats=(
    df_stats.set_index('school', append=True)
            .reset_index(level=[1], drop=True)
)
df_stats_meta = df_stats[['conference','homeAway','points']]
df_stats=(
    pd.concat([pd.DataFrame(x) for x in df_stats.stats],
              keys=df_stats.index,
              names=df_stats.index.names)
    .set_index('category',append=True)
    .reset_index(level=[2], drop=True)
    .unstack(level=2)
)

df_stats.columns = df_stats.columns.get_level_values(1)
df_stats.columns.name = None
df_stats = df_stats.join(df_stats_meta)

df_stats

# COMMAND ----------

#join lines_fixed into df_stats

spread_cols=[
        'spread',
        'spreadOpen',
        'overUnder',
        'overUnderOpen',
        'homeMoneyline',
        'awayMoneyline',
        'actual_spread',
        'B',
        'BW'
]

lines_fixed_by_provider = (lines_fixed[spread_cols].unstack(level=1))

lines_fixed_temp=lines_fixed.xs(key='consensus', level=1)
df_stats=df_stats.join(lines_fixed_temp, on='id')

# COMMAND ----------

#apply line correction, copy unaltered for debug

#df_stats_uncorrected=df_stats.copy()

#df_stats['isaway'] = df_stats.apply(lambda x: 1 if x['homeAway']=='away' else -1, axis=1)
# df_stats[spread_cols]=(df_stats[spread_cols+['isaway']]
#                        .apply(lambda x: x*x['isaway'], axis=1)
#                        .drop('isaway', axis=1))
df_stats.head()

# COMMAND ----------

df_stats.spread = df_stats.spread.replace('',np.NaN).astype(float)

def fix_lines_away(row):
    # fixes away lines, since lines are always given based on home team
    x = row.copy()
    adj_list = ['spread', 'spreadOpen', 'B', 'BW', 'actual_spread']
    for n in adj_list:
        try:
            x.loc[n]*=-1
        except TypeError:
            pass
    return x

df_stats = df_stats.apply(lambda x: fix_lines_away(x) if x.loc['homeAway']=='away' else x, axis=1)

df_stats.to_csv('df_stats_2021.csv')

# COMMAND ----------

df_stats_feed = df_stats[df_stats['week']<7]
df_stats_wk7 = df_stats[df_stats['week']==7]
df_stats_wk7 = df_stats_wk7.set_index('season', append=True)
df_stats_feed = df_stats_feed.set_index('season', append=True)
df_stats_feed.describe()
df_stats_wk7.describe()

# COMMAND ----------

df_stats_feed.columns

# COMMAND ----------

gameId = '401112224'

test_df=pd.DataFrame(pd.read_json(test_request.content).teams.iloc[0])
test_df=pd.concat([pd.DataFrame(x) for x in test_df.stats], keys=test_df.school)

test_df = (test_df.set_index('category',append=True)
                  .reset_index(level=1,drop=True)
                  .unstack(level=1))

test_df.index = pd.MultiIndex.from_tuples([(gameId, x) for x in test_df.index.values])
test_df.columns = [x[1] for x in test_df.columns]
test_df

# COMMAND ----------

lines_fixed[['week','homeTeam','homeScore','awayScore','actual_spread','spread','B']]

# COMMAND ----------

import numpy as np
from scipy import stats
np.random.seed(12345678)
x = stats.norm.rvs(loc=5, scale=3, size=100)
y = stats.norm.rvs(loc=5, scale=3, size=1000)
stats.shapiro(y)
