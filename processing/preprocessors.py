from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


features_transformer = ColumnTransformer([('ot_encoder',
                                           OneHotEncoder(handle_unknown='ignore'),
                                           ['day'])],
                                         remainder='passthrough')
