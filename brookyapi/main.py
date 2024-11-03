import os
import logging

from flask import Flask, request
from functions import brooky # pylint: disable=import-error

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

@app.route('/')
def default_func():
    return 'Welcome to Brooky API!'


@app.route('/brooksai', methods=['POST'])
def brooksai_predict():
    return brooky.predict(request)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
