import glob
import pandas as pd
import random
import os
import shutil

dftr = pd.read_csv('train.csv')
dfvl = pd.read_csv('val.csv')
dfts = pd.read_csv('test.csv')

datas = {}

for k, v in dftr.iterrows():
    if v['label'] in datas:
        datas[v['label']].append(v['filename'])
    else:
        datas[v['label']] = [v['filename']]

for k, v in dfvl.iterrows():
    if v['label'] in datas:
        datas[v['label']].append(v['filename'])
    else:
        datas[v['label']] = [v['filename']]

for k, v in dfts.iterrows():
    if v['label'] in datas:
        datas[v['label']].append(v['filename'])
    else:
        datas[v['label']] = [v['filename']]

for k, v in datas.items():
    assert len(v) == 600
    random.shuffle(v)

    if not os.path.exists(os.path.join('train_', k)):
        os.mkdir(os.path.join('train_', k))

    if not os.path.exists(os.path.join('test_', k)):
        os.mkdir(os.path.join('test_', k))

    for fname in v[:-100]:
        shutil.copy(os.path.join('images', fname), os.path.join('train_', k, fname))

    for fname in v[-100:]:
        shutil.copy(os.path.join('images', fname), os.path.join('test_', k, fname))