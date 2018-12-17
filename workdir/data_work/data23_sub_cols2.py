import datetime
import numpy as np

from hyperopt import hp, fmin, tpe
import os
import sys
sys.path.insert(0, os.getcwd())
import workdir.classes.config
from qml.cv import QCV
from qml.models import QXgb, QAvg, QAvgOneModelData
from workdir.classes.models import qm



cv = QCV(qm)

# model_id = qm.add_by_params(
#     QXgb(
# ** {"alpha": 1.0, "booster": "gbtree", "colsample_bylevel": 0.7, "colsample_bytree": 0.8, "eta": 0.004, "eval_metric": "logloss",
#     "gamma": 0.2, "max_depth": 4, "num_boost_round": 2015, "objective": "binary:logistic", "subsample": 0.8, "tree_method": "hist"}
#     ),
#     'hyperopt xgb', level=-1
# )

model_id =qm.add_by_params(QAvgOneModelData(416, 2), level=-2)

cv.features_sel_del(
    model_id, 23,
    early_stop_cv=lambda x: x<0.557, # minmax
    log_file='workdir/logs/data23_sub_cols3.txt',
    exclude=[
'data__daily__data_vol_mb__median__max',
     'data__daily__cell_count__min',
'voice__hourly__voice_dur_min__sum__max',
'voice__daily__voice_dur_min__sum__median',
'data__daily__data_vol_mb__sum__min',
'data__daily__data_vol_mb__avg__avg',
'voice__hourly__voice_dur_min__median__max',
'voice__hourly__voice_dur_min__sum__min',
'voice__daily__cell_count__median',
'voice__daily__voice_dur_min__sum__max',
'voice__cell_lac_id__count',
'data__daily__data_vol_mb__max__min',
'voice__voice_dur_min__days__count',
'voice__daily__voice_dur_min__avg__max',
'data__daily__cell_count__median',
'voice__hourly__cell_count__avg',
'voice__hourly__cell_count__median',
'data__hourly__data_vol_mb__avg__median',
'voice__hourly__voice_dur_min__sum__avg',
'data__hourly__data_vol_mb__median__max',
'data__hourly__data_vol_mb__avg__max',
'voice__voice_dur_min__avg',
'data__daily__data_vol_mb__sum__avg',
    ],
    columns=[


'voice__voice_dur_min__max',



'voice__daily__voice_dur_min__sum__min',
'voice__daily__cell_count__avg',
'voice__daily__voice_dur_min__avg__median',
'data__daily__data_vol_mb__median__avg',
'voice__voice_dur_min__sum',

'voice__hourly__voice_dur_min__median__avg',



'voice__hourly__cell_count__max',

'data__daily__data_vol_mb__avg__median',

'data__daily__cell_count__max',

'data__daily__data_vol_mb__median__min',



'voice__voice_dur_min__hours__count',
'voice__daily__voice_dur_min__max__avg',
'voice__hourly__voice_dur_min__median__median',
# 'voice__hourly__voice_dur_min__avg__max',
# 'voice__daily__voice_dur_min__max__min',
# 'data__daily__data_vol_mb__max__max',
# 'data__data_vol_mb__avg',
# 'data__hourly__data_vol_mb__median__min',
# 'data__cell_lac_id__count',
# 'data__hourly__cell_count__avg',
# 'data__hourly__data_vol_mb__max__median',
# 'data__data_vol_mb__sum',
# 'voice__daily__voice_dur_min__median__max',
# 'data__daily__data_vol_mb__sum__median',
# 'data__hourly__cell_count__min',
# 'voice__hourly__voice_dur_min__max__min',
# 'data__hourly__data_vol_mb__median__avg',
# 'voice__hourly__voice_dur_min__median__min',
# 'data__hourly__data_vol_mb__sum__min',
# 'data__daily__data_vol_mb__max__median',
# 'voice__daily__voice_dur_min__median__median',
# 'voice__daily__voice_dur_min__avg__avg',
# 'data__data_vol_mb__median',
# 'data__daily__data_vol_mb__sum__max',
# 'data__data_vol_mb__hours__count',
# 'data__daily__data_vol_mb__median__median',
# 'voice__hourly__cell_count__min',
# 'data__daily__cell_count__avg',
# 'data__hourly__cell_count__median',
# 'data__hourly__data_vol_mb__sum__avg',
# 'voice__daily__voice_dur_min__max__median',
# 'voice__hourly__voice_dur_min__max__avg',
# 'data__daily__data_vol_mb__avg__min',
# 'data__daily__data_vol_mb__max__avg',
# 'data__hourly__data_vol_mb__sum__max',
# 'data__data_vol_mb__max',
# 'voice__hourly__voice_dur_min__max__max',
# 'voice__hourly__voice_dur_min__sum__median',
# 'voice__hourly__voice_dur_min__max__median',
# 'data__hourly__data_vol_mb__avg__avg',
# 'data__data_vol_mb__days__count',
# 'voice__daily__cell_count__min',
# 'data__hourly__data_vol_mb__max__avg',
# 'voice__daily__voice_dur_min__median__min',
# 'voice__hourly__voice_dur_min__avg__median',
# 'voice__daily__voice_dur_min__avg__min',
# 'data__hourly__data_vol_mb__avg__min',
# 'data__hourly__data_vol_mb__max__max',
# 'voice__hourly__voice_dur_min__avg__avg',
# 'voice__daily__voice_dur_min__max__max',
# 'data__hourly__data_vol_mb__median__median',
# 'data__hourly__data_vol_mb__sum__median',
# 'voice__voice_dur_min__median',
# 'data__hourly__data_vol_mb__max__min',
# 'data__hourly__cell_count__max',
# 'voice__hourly__voice_dur_min__avg__min',
# 'voice__daily__voice_dur_min__median__avg',
# 'voice__daily__voice_dur_min__sum__avg',
# 'data__daily__data_vol_mb__avg__max',
# 'voice__daily__cell_count__max',

    ]
)


