import json
import math
import struct

import numpy as np


PROTOCOL = "ridex_hpsv2_float32_v1"
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
LENGTH = struct.Struct("!I")


def encode_batch(images, prompts):
    images = np.ascontiguousarray(images, dtype="<f4")
    header = json.dumps({"shape": images.shape, "prompts": list(prompts)}).encode()
    total = LENGTH.size + len(header) + images.nbytes
    if len(header) > MAX_HEADER_BYTES or total > MAX_BODY_BYTES:
        raise ValueError("HPSv2 batch exceeds the request size limit")
    return LENGTH.pack(len(header)) + header + images.tobytes()


def decode_batch(body):
    if not LENGTH.size < len(body) <= MAX_BODY_BYTES:
        raise ValueError("Invalid HPSv2 body size")
    (length,) = LENGTH.unpack_from(body)
    start = LENGTH.size + length
    if not 0 < length <= MAX_HEADER_BYTES or start >= len(body):
        raise ValueError("Invalid HPSv2 header size")
    metadata = json.loads(body[LENGTH.size:start])
    if not isinstance(metadata, dict):
        raise ValueError("Invalid HPSv2 header")
    shape, prompts = metadata.get("shape"), metadata.get("prompts")
    if (not isinstance(shape, list) or len(shape) != 4 or
        any(type(n) is not int or not 1 <= n <= 4096 for n in shape) or shape[1] not in (3, 4)):
        raise ValueError("Expected an NCHW RGB(A) batch")
    if not isinstance(prompts, list) or len(prompts) != shape[0] or any(not isinstance(p, str) for p in prompts):
        raise ValueError("Expected one text prompt per image")
    if len(body) - start != 4 * math.prod(shape):
        raise ValueError("Invalid HPSv2 tensor size")
    images = np.frombuffer(body, dtype="<f4", offset=start).reshape(shape).astype(np.float32, copy=True)
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Expected finite image values in [0, 1]")
    return images, prompts
