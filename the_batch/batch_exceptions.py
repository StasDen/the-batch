"Defines custom exception specific to The Batch app."

from exceptions.base import BaseException


class BatchLLMAnswerGenerationError(BaseException): ...


class BatchLLMInitializationError(BaseException): ...
