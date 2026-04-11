from redis import Redis
from rq import Connection, Worker

from api.app.settings import settings


def run_trainer_worker() -> None:
    redis = Redis.from_url(settings.redis_url)
    with Connection(redis):
        worker = Worker(["trainer"])
        worker.work(with_scheduler=True)


if __name__ == "__main__":
    run_trainer_worker()

