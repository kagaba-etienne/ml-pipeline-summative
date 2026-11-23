"""Flood the API with prediction requests (through nginx, so load is spread over the API containers).

    uv run locust --host http://localhost:8080 --headless -u 50 -r 5 -t 2m --csv loadtest/api1

Each simulated user sends a real leaf image from data/test to /predict (10x more often than
/health) and waits 0.5-1.5 s between requests.
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

# 5 test images per class, read once into memory
IMAGES = [(p.name, p.read_bytes())
          for d in sorted(Path("data/test").iterdir())
          for p in sorted(d.iterdir())[:5]]


class FarmerUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        self.client.post("/api/predict", files={"file": (name, data, "image/jpeg")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/api/health", name="/health")
