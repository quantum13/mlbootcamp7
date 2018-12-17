from qml import config

config.QML_DATA_DIR = '/data/ml/telecom/workdir/data/'
config.QML_DB_CONN_STRING = 'postgresql://postgres:rootpassword@localhost/telecom'

config.QML_TRAIN_X_FILE_MASK = 'workdir/data/v{0:0=4d}_train_x.csv'
config.QML_TRAIN_Y_FILE_MASK = 'workdir/data/train_y.csv'
config.QML_TEST_X_FILE_MASK = 'workdir/data/v{0:0=4d}_test_x.csv'

config.QML_RES_COL = 'csi'
config.QML_RES_COL2 = ''

config.QML_INDEX_COL = 'sk_id'
