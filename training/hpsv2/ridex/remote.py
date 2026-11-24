import math
import os
import time

import requests
import torch

from ridex.reward_protocol import PROTOCOL, encode_batch


def canonical_remote():
    urls = [u.strip().rstrip("/") for u in os.environ.get("HPSV2_REWARD_URLS", "").split(",") if u.strip()]
    if not urls:
        raise ValueError("Set HPSV2_REWARD_URLS to a RiDEx HPSv2 service")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    url = urls[rank % len(urls)] + "/score"
    batch_size = int(os.environ.get("HPSV2_REWARD_BATCH_SIZE", "8"))
    if batch_size < 1:
        raise ValueError("HPSV2_REWARD_BATCH_SIZE must be positive")
    session = requests.Session()
    session.trust_env = False

    def score(images, prompts):
        values = images.detach().to(dtype=torch.float32, device="cpu").numpy()
        if len(values) != len(prompts):
            raise ValueError("Expected one prompt per image")
        scores = []
        for start in range(0, len(prompts), batch_size):
            stop = min(start + batch_size, len(prompts))
            body = encode_batch(values[start:stop], prompts[start:stop])
            for attempt in range(3):
                try:
                    response = session.post(url, data=body, timeout=(5, 120),
                                            headers={"Content-Type": "application/octet-stream"})
                    if response.status_code in {429, 500, 502, 503, 504}:
                        raise requests.ConnectionError(f"HPSv2 service status {response.status_code}")
                    break
                except (requests.ConnectionError, requests.Timeout):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            response.raise_for_status()
            payload = response.json()
            batch = [float(x) for x in payload["scores"]]
            if payload["reward_protocol"] != PROTOCOL or len(batch) != stop-start or not all(map(math.isfinite, batch)):
                raise ValueError("Invalid HPSv2 service response")
            scores.extend(batch)
        return scores

    return score
