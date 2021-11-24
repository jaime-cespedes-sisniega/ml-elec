from ml_pipeline.base_pipeline import BasePipeline
from ml_pipeline.preprocessors import features_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class ModelPipeline(BasePipeline):

    def __init__(self, random_state):
        self.pipeline = Pipeline(
            steps=[('transformer', features_transformer),
                   ('clf', RandomForestClassifier(random_state=random_state,
                                                  class_weight='balanced'))])
        self.target_encoder = LabelEncoder()

    def fit(self, X, y):
        y_encoded = self.target_encoder.fit_transform(y)
        self.pipeline.fit(X, y_encoded)

    def predict(self, X):
        pred_encoded = self.pipeline.predict(X)
        pred = self.target_encoder.inverse_transform(pred_encoded)
        return pred
