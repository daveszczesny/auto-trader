import os
import logging

from flask import Flask, request
from functions import brooky

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)


model = None
lstm_states = None

@app.route('/')
def defaul_func():
    
    name = os.environ.get('NAME', 'World')
    return f'Hello, {name}!'


@app.route('/brooksai', methods=['POST'])
def brooksai_predict():
    brooky.predict(request)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))