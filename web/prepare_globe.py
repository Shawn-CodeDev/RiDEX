"""Create native-aspect thumbnails for the interactive gallery."""
import json
import os
from pathlib import Path
import socket

from PIL import Image


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('Run thumbnail preparation on an allocated compute node.')
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / 'assets/comparison_manifest.json').read_text())['examples']
    destination = root / 'assets/globe'
    destination.mkdir(exist_ok=True)
    total = 0
    for row in rows:
        with Image.open(root / row['images']['RiDEx']) as original:
            thumbnail = original.convert('RGB')
            thumbnail.thumbnail((384, 384), Image.Resampling.LANCZOS)
            output = destination / f"{row['id']}.webp"
            thumbnail.save(output, format='WEBP', quality=83, method=6)
            total += output.stat().st_size
    print(f'{len(rows)} native-aspect thumbnails; {total / 1024:.0f} KiB total')


if __name__ == '__main__':
    main()
