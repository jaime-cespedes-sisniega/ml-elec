from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from processing.preprocessors import features_transformer


class ModelPipeline:

    def __init__(self, random_state):
        self.pipeline = Pipeline(steps=[('transformer',
                                         features_transformer),
                                        ('clf',
                                         RandomForestClassifier(random_state=random_state))])
        self.target_encoder = LabelEncoder()

    def fit(self, X, y):
        y_encoded = self.target_encoder.fit_transform(y)
        self.pipeline.fit(X, y_encoded)

    def predict(self, X):
        pred = self.pipeline.predict(X)
        return pred

    def transform_target(self, y):
        y_encoded = self.target_encoder.transform(y)
        return y_encoded
