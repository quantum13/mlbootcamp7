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
    model_id, 16,
    early_stop_cv=lambda x: x<0.53, # minmax
    log_file='workdir/logs/data16_sub_cols1.txt',
    exclude=[
'sum_minutes__avg__mon2',
'rent_channel__max',
'sum_data_mb__max__mon3',
'com_cat_33__median',
'com_cat_31__max',
'com_cat_2__last__cat_23',
        'com_cat_1__cat_8',

    ],
    columns=[


'com_cat_2__last__cat_22',
'sum_data_min__max__mon1',
'cell_lac_count__mon2',
'com_cat_23__median',
'com_cat_3__17',
'com_cat_3__16',
'sum_data_min__avg__mon3',
'com_cat_23__min',
'com_cat_18__median',
'sum_data_min__avg',
'revenue__sum',
'sum_minutes__median__mon1',
'com_cat_22__max',
'com_cat_2__last__cat_55',
'com_cat_2__last__cat_54',
'com_cat_2__last__cat_53',
'com_cat_19__min',
'com_cat_27__sum',
'roam__max',
'rent_channel__sum',
'com_cat_7__2',
'com_cat_27__min',
'internet_type_id__null',
'cost__median',
'com_cat_17__median',
'com_cat_19__sum',
'com_cat_19__max',
'rent_channel__avg',
'com_cat_18__max',
'vas__median',
'com_cat_18__min',
'com_cat_18__sum',
'com_cat_22__avg',
'roam__min',
'com_cat_30__median',
'roam__avg',
'sum_minutes__median__mon3',
'com_cat_19__avg',
    ]
)


