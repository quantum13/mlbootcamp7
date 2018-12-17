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
# print(cv.cross_val(465, 15))
# print(cv.cross_val(465, 16))
# print(cv.cross_val(466, 15))
# print(cv.cross_val(466, 16))
new_model_id = qm.add_by_params(QAvgOneModelData(636, 8), level=-2)
print(new_model_id)
qm.qpredict(new_model_id, 25)
# # print(new_model_id)
#
# for model_id in [416]: #450
#     for data_id in [25]:#13,15,16,17
#
#         for i in range(8):
#             res = cv.cross_val(model_id, data_id, seed=1000 + i, force=True)
#             print('##', model_id, data_id, i,  res)
