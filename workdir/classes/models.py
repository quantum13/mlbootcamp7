from qml.models import QModels, QXgb
from sklearn.linear_model import LogisticRegression

qm = QModels()


qm.add(1,
    QXgb(
        booster='gbtree',
        objective='binary:logistic',
        eval_metric='auc',
        eta=0.005,#learn_rate
        max_depth=3,

        num_boost_round=1000

    ),
    'simple xgb linear'
)


qm.add(2,
    QXgb(
        booster='gblinear',
        objective='binary:logistic',
        eval_metric='auc',
        subsample=0.5,
        eta=0.1,#learn_rate
        max_depth=3,

        num_boost_round=100

    ),
    'simple xgb linear'
)




qm.add(1000,
    QXgb(),
    'dummy'
)
