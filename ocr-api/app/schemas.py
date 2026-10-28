from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from .errors import APIError
import json


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_json(value):
    return json.loads(value, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["spotting", "ocr", "formula", "table"]
    max_new_tokens: StrictInt = Field(default=4096, ge=1, le=4096)


def parse_payload(value, limit):
    try:
        payload = Payload.model_validate(parse_json(value))
        if payload.max_new_tokens > limit:
            raise ValueError("requested output exceeds server limit")
        return payload
    except (ValueError, TypeError) as exc:
        raise APIError(400, "BAD_REQUEST", "invalid task payload") from exc


def control_body(body):
    try:
        if body and parse_json(body) != {}:
            raise ValueError("expected empty object")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise APIError(400, "BAD_REQUEST", "control body must be empty or {}") from exc
