
from flask import jsonify, make_response
from google.cloud import storage

from utils.exceptions import ErrorEntry
from utils.constants import StatusCode


def get_bucket(bucket_name):
    """
    Get the bucket object for the given bucket name.

    :param bucket_name: The name of the bucket.
    :return: The bucket object.
    """
    storage_client = storage.Client()
    return storage_client.get_bucket(bucket_name)


def handle_error(error: ErrorEntry, status_code: StatusCode):
    """
    This method will handle errors and return a response
    """

    response = jsonify({'error': error.to_dict()})
    return make_response(response, status_code)
