import joblib


def load_model(path):
    model = joblib.load(path)
    return model
