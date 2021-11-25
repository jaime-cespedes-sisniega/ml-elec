from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


features_transformer = ColumnTransformer([
    ('ot_encoder',
     OneHotEncoder(handle_unknown='ignore'),
     [0])],
    remainder='passthrough')
