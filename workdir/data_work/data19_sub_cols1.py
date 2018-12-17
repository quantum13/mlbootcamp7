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
    model_id, 19,
    early_stop_cv=lambda x: x<0.557, # minmax
    log_file='workdir/logs/data19_sub_cols1.txt',
    exclude=[
    ],
    columns=[
'com_cat_2__last__cat_45',
'itc__last',
'com_cat_2__last__cat_44',
'com_cat_2__last__cat_43',
'com_cat_2__last__cat_42',
'com_cat_2__last__cat_41',
'com_cat_2__last__cat_40',
'com_cat_2__last__cat_39',
'com_cat_2__last__cat_38',
'com_cat_2__last__cat_37',
'com_cat_2__last__cat_36',
'arpu_group__04',
'com_cat_23__last',
'com_cat_21__last',
'com_cat_30__max',
'com_cat_29__median',
'com_cat_2__last__cat_51',
'com_cat_2__last__cat_50',
'com_cat_2__last__cat_49',
'com_cat_2__last__cat_48',
'com_cat_2__last__cat_47',
'com_cat_2__last__cat_46',
'sum_data_min__avg__mon1',
'com_cat_26__sum_zero',
'com_cat_26__2',
'com_cat_26__1',
'com_cat_25__sum_zero',
'com_cat_24__sum_not3112',
'com_cat_2__last__cat_9',
'com_cat_2__last__cat_13',
'com_cat_2__last__cat_12',
'com_cat_2__last__cat_11',
'com_cat_2__last__cat_10',
'com_cat_22__sum',
'com_cat_2__last__cat_7',
'com_cat_2__last__cat_6',
'com_cat_2__last__cat_5',
'com_cat_2__last__cat_4',
'com_cat_2__last__cat_3',
'com_cat_2__last__cat_2',
'com_cat_2__last__cat_1',
'sum_data_min__avg__mon2',
'com_cat_3__10',
'com_cat_27__last',
'com_cat_34__last',
'com_cat_2__last',
'com_cat_3__09',
'arpu_group__02',
'arpu_group__03',
'arpu_group__10',
'sum_minutes__sum__mon3',
'com_cat_33__sum',
'com_cat_2__last__cat_20',
'com_cat_2__last__cat_19',
'com_cat_2__last__cat_18',
'com_cat_2__last__cat_17',
'com_cat_2__last__cat_16',
'com_cat_2__last__cat_15',
'device_type_id__4',
'cell_lac_count',
'com_cat_2__last__cat_21',
'com_cat_7__8',
'com_cat_3__last__cat_7',
'arpu_group__last__cat_4.0',
'rent_channel__median',
'com_cat_31__median',
'com_cat_3__last__cat_15',
'com_cat_28__sum',
'com_cat_1__cat_4',
'com_cat_3__last__cat_9',
    ]
)


