from redis import Redis
from rq import Worker

from api.app.settings import settings


def run_trainer_worker() -> None:
    redis = Redis.from_url(settings.redis_url)
    worker = Worker(["trainer"], connection=redis)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    run_trainer_worker()
