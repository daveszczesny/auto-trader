from dataclasses import dataclass
from dataclasses_json import LetterCase, dataclass_json


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ErrorEntry:
    code: str
    message: str


class ErrorSet:
    INVALID_INPUT = ErrorEntry(
        code='invalid.input.missing.required.fields',
        message='One or more required fields are missing'
    )
    INVALID_OBSERVATION_LENGTH = ErrorEntry(
        code='invalid.input.observation.length',
        message='Invalid observation length. Observation must contain 9 elements'
    )

    MODEL_NOT_FOUND = ErrorEntry(
        code='model.not.found',
        message='Model file not found'
    )

    MODEL_FORBIDDEN = ErrorEntry(
        code='model.forbidden',
        message='Permission denied to download model'
    )

    MODEL_CONFLICT = ErrorEntry(
        code='model.conflict',
        message='Failed to download model due to conflict error'
    )

    INSTANCE_NOT_FOUND = ErrorEntry(
        code='instances.not.found',
        message='Instances file not found'
    )

    INSTANCES_FORBIDDEN = ErrorEntry(
        code='instances.forbidden',
        message='Permission denied to download or update instances'
    )

    INSTANCES_CONFLICT = ErrorEntry(
        code='instances.conflict',
        message='Failed to download or update instances due to conflict error'
    )

    MODEL_NOT_WARMED_UP = ErrorEntry(
        code='model.not.warmed.up',
        message='Model has not been warmed up. Call warmup endpoint first'
    )

    UNKNOWN_ERROR = ErrorEntry(
        code='unknown.error',
        message='Unknown error occurred'
    )
