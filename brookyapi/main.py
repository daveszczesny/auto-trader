import os
import logging

from flask import Flask
from endpoints.brooksai import brooksai_bp

app = Flask(__name__)

app.register_blueprint(brooksai_bp)

logging.basicConfig(level=logging.INFO)

@app.route('/')
def default_func():
    return 'Welcome to Brooky API!'


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
