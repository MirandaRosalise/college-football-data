# Databricks notebook source
import pandas as pd
import numpy as np

df_stats = pd.read_csv('../output/df_stats.csv')
df_stats.head()

df_stats_2021 = pd.read_csv('../output/df_stats_2021.csv')
df_stats = pd.concat([df_stats, df_stats_2021], axis=0)
df_stats

# COMMAND ----------

#column list

print(df_stats.columns)
print(df_stats.dtypes)

# COMMAND ----------



# COMMAND ----------

# this data is dirty! less than 15% of rows contain all the stats...
# skip imputation or drop, we will work around this in model selection

filter_cols = [
    x for x in df_stats.columns if x not in ['spreadOpen',
                                             'overUnder',
                                             'overUnderOpen',
                                             'homeMoneyline',
                                             'awayMoneyline']
]
df_stats.dropna(subset=filter_cols).count().id/df_stats.count().id*1.0


# COMMAND ----------

# feature processing of offensive stats

df = df_stats.copy()
df = df.set_index(['school','season'])

#df = df.dropna(subset=['completionAttempts', 'homeAway'])
df.B = df.homeScore-df.awayScore+df.spread
df.B = df.apply(
    lambda x: x.loc['homeScore'] - x.loc['awayScore'] + x.loc['spread'] 
        if x.loc['homeAway']=='home'
    else x.loc['awayScore'] - x.loc['homeScore'] + x.loc['spread'] 
        if x.loc['homeAway']=='away'
    else 0,
                axis=1
               )
df['completionAttempts'] = df.completionAttempts.str.split('-')
df['completions'] = (
    df.completionAttempts.apply(lambda x: x[0] if type(x) is list else x).astype(int, errors='ignore')
)
df['attempts'] = (
    df.completionAttempts.apply(lambda x: x[1] if type(x) is list else x).astype(int, errors='ignore')
)
df['completionPercentage'] = (
    df.apply(lambda x: x.loc['completions']/
             x.loc['attempts']*1.0 
             if type(x.loc['completions'])==int
            and type(x.loc['attempts'])==int
            else np.NaN,
            axis=1)
)
df['fourthDownEff'] = df.fourthDownEff.str.split('-')
df['fourthDownEff'] = (
    df.fourthDownEff.apply(lambda x: int(x[0]) if type(x) is list else x)/
    df.fourthDownEff.apply(lambda x: float(x[1]) if type(x)==list else x)*1.0
)
df['thirdDownEff'] = df.thirdDownEff.str.split('-')
df['thirdDownEff'] = (
    df.thirdDownEff.apply(lambda x: int(x[0]) if type(x) is list else x)/
    df.thirdDownEff.apply(lambda x: float(x[1]) if type(x)==list else x)*1.0
)
df['totalPenaltiesYards'] = df.totalPenaltiesYards.str.split('-')
df['totalpenalties'] = (df.totalPenaltiesYards
                        .apply(lambda x: x[0] if type(x) is list else x)
                        .astype(int, errors='ignore'))
df['penaltyYards'] = (df.totalPenaltiesYards
                      .apply(lambda x: x[1] if type(x) is list else x)
                      .apply(lambda x: 0 if type(x) is list and x[0]=='0' else x)
                      .apply(lambda x: np.nan if x=='' else x)
                      .astype(int, errors='ignore')
                     )
df['possessionTime'] = df.possessionTime.str.split(':')
df['possessionTime'] = df.possessionTime.apply(lambda x: int(x[0])*60+int(x[1]) if type(x) is list else np.nan)

feature_list_excl=[
    'id',
    'completionAttempts',
    'conference',
    'homeAway',
    #'seasonType',
    'homeTeam',
    'awayTeam',
    'homeConference',
    'awayConference',
    'lines',
    'formattedSpread',
    'totalPenaltiesYards'
]

df = df.drop(feature_list_excl, axis=1)
df_fut = df.xs(2021.0, level=1)
df = df.drop(2021.0, level=1)

# COMMAND ----------

df.index.levels[1]

# COMMAND ----------

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

curweek = 10

pred = df[df.week==curweek].drop('week', axis=1).B
df_feed = df[(df.week<curweek) & (df.week>(curweek-5))].drop('week', axis=1)
df_feed = df_feed.loc[pred.index.intersection(df_feed.index)]
pred = pred.loc[pred.index.intersection(df_feed.index)]
pred = pred.apply(lambda x: -1 if x<=0 else 1)

df_feed = df_feed.groupby(level=[0,1]).median()

X_train, X_test, Y_train, Y_test = train_test_split(df_feed, pred, train_size=0.8)

# COMMAND ----------

pred.groupby(pred>0).count()

# COMMAND ----------

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier as hgbc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score

clf = hgbc()
clf.fit(X_train, Y_train)

print(precision_score(Y_test, clf.predict(X_test), average=None))

print(balanced_accuracy_score(Y_test, clf.predict(X_test)))

print(compute_class_weight('balanced', [-1, 1], y=Y_test.values))

# COMMAND ----------

df_feed_fut = df_fut[df_fut.week<curweek].drop('week', axis=1)
df_feed_fut = df_feed_fut.groupby(level=0).median()

df_cur_pred = pd.DataFrame(index=df_feed_fut.index, data=clf.predict_proba(df_feed_fut))

# COMMAND ----------

import requests

api_key='redacted'
test_headers = {'Authorization': 'Bearer ' + api_key}

df_schedule = pd.read_json(requests.get('https://api.collegefootballdata.com/games', params={'year':2021},headers=test_headers).content)
df_schedule.head()
df_schedule = df_schedule[df_schedule['week']==curweek][['home_team', 'away_team']]
df_schedule = df_schedule.set_index(['home_team', 'away_team'])

# COMMAND ----------

df_bets = df_schedule.copy()

df_bets = df_bets.assign(home_0=df_bets.join(df_cur_pred[0], on='home_team'),
               home_1=df_bets.join(df_cur_pred[1], on='home_team'),
               away_0=df_bets.join(df_cur_pred[0], on='away_team'),
               away_1=df_bets.join(df_cur_pred[1], on='away_team')
              )

#df_bets['pick'] = df_bets.apply(lambda x: x.loc[] if (x.loc['home_0'] < x.loc['away_0']) else x.loc['away_team'], axis=1)
df_bets['diff_0'] = abs(df_bets.home_0-df_bets.away_0)
df_bets['diff_1'] = abs(df_bets.home_1-df_bets.away_1)
df_bets['hlatnp'] = df_bets.home_0>df_bets.away_0

df_bets.sort_values(by='diff_0', ascending=False).head(24)

# COMMAND ----------

df_test = pd.concat([Y_test, pd.Series(clf.predict(X_test), index=Y_test.index)], axis=1)
df_test = df_test.join(df[df.week==7][['homeAway','homeScore', 'awayScore', 'spread', 'actual_spread']].reindex(df_test.index))
df_test

# COMMAND ----------

# MAGIC %md
# MAGIC ### Todo:
# MAGIC * matchups -> predict result from clf scores
