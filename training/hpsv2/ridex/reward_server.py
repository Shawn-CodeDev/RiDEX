import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from ridex.reward_protocol import MAX_BODY_BYTES, PROTOCOL, decode_batch


class RewardHandler(BaseHTTPRequestHandler):
    def send_json(self, status, obj):
        data = json.dumps(obj, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/health":
            return self.send_json(404, {"error": "Unknown endpoint"})
        self.send_json(200, {"ok": True, "reward_type": "hpsv2", "reward_protocol": PROTOCOL})

    def do_POST(self):
        if self.path != "/score":
            return self.send_json(404, {"error": "Unknown endpoint"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY_BYTES:
                raise ValueError("Invalid request size")
            self.connection.settimeout(30)
            images, prompts = decode_batch(self.rfile.read(size))
        except (ValueError, OSError) as exc:
            return self.send_json(400, {"error": str(exc)})
        try:
            with torch.no_grad():
                scores = self.server.scorer(torch.from_numpy(images), prompts).detach().float().cpu().tolist()
            if len(scores) != len(prompts) or not all(map(math.isfinite, scores)):
                raise ValueError("Invalid scorer output")
        except (RuntimeError, ValueError) as exc:
            print(f"HPSv2 scoring failed: {exc}", flush=True)
            return self.send_json(500, {"error": "HPSv2 scoring failed"})
        self.send_json(200, {"scores": scores, "reward_protocol": PROTOCOL})


def main():
    from flow_grpo.hpsv2_scorer import HPSv2Scorer

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openclip-checkpoint", required=True)
    parser.add_argument("--hps-checkpoint", required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible to the HPSv2 service")
    scorer = HPSv2Scorer(device=args.device, dtype=torch.float32,
                         openclip_checkpoint=args.openclip_checkpoint, hps_checkpoint=args.hps_checkpoint)
    with HTTPServer((args.host, args.port), RewardHandler) as server:
        server.scorer = scorer
        print(f"HPSv2 listening on http://{args.host}:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
