# Databricks notebook source
import pandas as pd
import numpy as np
import requests
import json
import tensorflow as tf

url = 'https://api.collegefootballdata.com/'
api_key='redacted'
auth = {'Authorization': 'Bearer ' + api_key}

# COMMAND ----------

test_year='2021'
test_week='6'
test_team='Alabama'

params = {
    'year': test_year,
    'week': test_week,
    'team': test_team
}

r = requests.get(url+'plays', params=params, headers=auth)
df = pd.read_json(r.content)
df.head()

# COMMAND ----------

years_list = range(2015, 2020)
weeks_list = range(1, 17)

# fires 80 api requests w/o limiting
response_list = [
    requests.get(url=url + 'plays',
                 headers=auth,
                 params={
                     'year': y,
                     'week': w
                 })
    for y in years_list for w in weeks_list
]

# COMMAND ----------

# writes json
# i make no guarantee that this works as expected ;_;

import os
import math

path = '../input/'

for i in range(0, len(response_list)):
    filename = str(years_list[math.floor(i/16.0)]) + '_week_' + str(weeks_list[i % 16]) + '.json'
    filename = os.path.join(path, filename)
    with open(filename, 'w') as f:
        f.write(response_list[i].content.decode())


# COMMAND ----------

import math

df = pd.concat(
    [
        pd.read_json(response_list[i].content)
          .assign(
              year=years_list[math.floor(i/16.0)],
              week=weeks_list[i % 16]
          ) 
        for i in range(0, len(response_list))
    ]
)
df = df.dropna(subset=['play_text'])
df['margin'] = abs(df.offense_score-df.defense_score)
df['is_garbage'] = (
    df.apply(
        lambda x: 1 if (
        (x.loc['period']==2 and x.loc['margin']>=38) or
        (x.loc['period']==3 and x.loc['margin']>=28) or
        (x.loc['period']==4 and x.loc['margin']>=22)
        ) else 0,
        axis=1
    )
)
df['includes_penalty'] = df.apply(lambda x: 'penalty' in x.loc['play_text'].lower(), axis=1)
keep_cols = [
    'game_id',
    'drive_id',
    'id',
    'year',
    'week',
    'play_number',
    'period',
    'offense',
    'defense',
    'is_garbage',
    'includes_penalty',
    'clock',
    'offense_timeouts',
    'defense_timeouts',
    'yards_to_goal',
    'down',
    'distance',
    'scoring',
    'yards_gained',
    'play_type',
    'ppa'
]

df = df[keep_cols].set_index(['game_id', 'drive_id','id'])
df.clock = df.apply(lambda x: x.loc['clock']['minutes']*60+x.loc['clock']['seconds'], axis=1)
df.clock = df.apply(lambda x: x.loc['clock']+900 if (x.loc['period']==1 or x.loc['period']==3) else x.loc['clock'], axis=1)

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md
# MAGIC ## For the purposes of scoring, how different is a run from a pass?

# COMMAND ----------

# vectorize a 1000-play sample from each offense
fbs_filter = df.groupby('offense').count().query('ppa>3000').index
df.play_type = df.play_type.replace({
    'Passing Touchdown': 'Pass Reception',
    'Rushing Touchdown': 'Rush',
    'Kickoff Return Touchdown': 'Kickoff',
    'Punt Return Touchdown': 'Punt'
    }
)
df_offense = df[df.play_type.isin([
    'Pass Reception',
    'Pass Incompletion',
    'Rush',
    'Kickoff',
    'Punt'
])]
df_offense = df_offense.dropna(subset=['offense_timeouts', 'defense_timeouts'])
df_offense.ppa = df_offense.ppa.fillna(0)
df_offense = pd.get_dummies(df_offense, columns=['play_type'])
df_offense = df_offense[df_offense.offense.isin(fbs_filter)].drop(['defense', 'play_number'], axis=1)

df_offense.groupby(['offense', 'year']).count().period

# COMMAND ----------


df_offense=df_offense.groupby(['offense', 'year']).sample(600, random_state=69, replace=True)
#df_offense['idx'] = df_offense.groupby(['offense', 'year']).cumcount()
#df_offense_v = df_offense.set_index(['year', 'week'], append=True)
#df_offense_v = df_offense.pivot(index=['offense', 'year'], columns='idx')

# COMMAND ----------

df_offense

# COMMAND ----------



# COMMAND ----------

#apply pca

from sklearn.decomposition import PCA

oemb = PCA(n_components=127)
oemb.fit_transform(df_offense_v)
print(oemb)
