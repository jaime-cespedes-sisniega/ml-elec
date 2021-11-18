from datetime import datetime
from pathlib import Path
from typing import Dict

from api.data_models import Request, Response
from api.utils import load_model
from fastapi import FastAPI


api = FastAPI()

model = load_model(path=Path('api', 'models',
                             'model_pipeline.joblib'))


@api.post('/',
          response_model=Response,
          status_code=200)
def predict(data: Request):
    timestamp = datetime.today().isoformat()
    pred = _make_prediction(data)
    response = {'timestamp': timestamp,
                'class_pred': pred}
    return response


def _make_prediction(data):
    pred_encoded = model.predict([[*data.dict().values()]])
    pred = model.inverse_transform_target(y_encoded=pred_encoded)[0]
    return pred


@api.get('/health')
async def health() -> Dict[str, str]:
    """Health check function

    :return: Health check dict
    :rtype: Dict[str: str]
    """
    return {'Status': 'Ok!'}
