# Databricks notebook source
import pandas as pd
import numpy as np
import requests
import yaml
import json


# COMMAND ----------

api_key='redacted'
test_headers = {'Authorization': 'Bearer ' + api_key}

url='https://api.collegefootballdata.com/game/box/advanced'

r = requests.get(url, params={'gameId': '401110723'}, headers=test_headers)

# COMMAND ----------


headers={'Authorization': 'Bearer ' + api_key}

for p in params_list:
    payload = {'year': p[0],
               'gameId': p[1]}
    responses.append(requests.get(url=url,
                                  params=payload,
                                  headers=headers))

# COMMAND ----------

pd.json_normalize(r_content['teams']['ppa'])

# COMMAND ----------

print(json.dumps(r_content, indent=True))

