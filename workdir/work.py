import datetime
import random

from hyperopt import hp, fmin, tpe
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import workdir.classes.config # loads local config
from qml.cv import QCV
from qml.helpers import get_engine
from qml.models import QXgb, QAvg, QRankedAvg, QStackModel, QPostProcessingModel, QRankedByLineAvg, QAvgOneModelData
from workdir.classes.models import qm



cv = QCV(qm)
# print(cv.cross_val(416, 2))
# print(cv.cross_val(416, 3))
# print(cv.cross_val(416, 5))
# print(cv.cross_val(416, 6))
qm.qpredict(465, 24)

# for model_id in [416,450]: # 416
#     for data_id in [13,15]:
#         new_model_id = qm.add_by_params(QAvgOneModelData(model_id, 4), level=-2)
#         print('##', model_id, data_id, new_model_id)
#         print(cv.cross_val(new_model_id, data_id))
