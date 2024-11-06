from flask import Blueprint, request
from functions.brooksai import brooky
from utils.constants import StatusCode

brooksai_bp = Blueprint('brooksai', __name__, url_prefix='/brooksai')

@brooksai_bp.route('/predict', methods=['POST'])
def predict():
    return brooky.predict(request)


@brooksai_bp.route('/warmup', methods=['POST'])
def warmup():
    return brooky.warmup(request)
