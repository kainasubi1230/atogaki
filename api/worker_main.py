from redis import Redis
from rq import Connection, Worker

from api.app.settings import settings


def run_worker(queue_names: list[str]) -> None:
    redis = Redis.from_url(settings.redis_url)
    with Connection(redis):
        worker = Worker(queue_names)
        worker.work(with_scheduler=True)


if __name__ == "__main__":
    run_worker(["default"])
