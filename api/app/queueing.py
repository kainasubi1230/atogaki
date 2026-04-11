from collections.abc import Callable
import uuid

from .settings import settings


class InlineJob:
    def __init__(self, func: Callable[..., None], *args):
        self.id = f"inline-{uuid.uuid4().hex}"
        func(*args)


def enqueue(queue_name: str, func: Callable[..., None], *args):
    if settings.rq_inline:
        return InlineJob(func, *args)
    from redis import Redis
    from rq import Queue

    redis = Redis.from_url(settings.redis_url)
    queue = Queue(queue_name, connection=redis)
    return queue.enqueue(func, *args, job_timeout=3600)
