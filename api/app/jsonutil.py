import json


def dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def loads(payload: str | None) -> dict:
    if not payload:
        return {}
    return json.loads(payload)

