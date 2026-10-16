"""Videos side by side, each scaled to the same height and labelled, cut to the shortest.

    python scripts/compare_videos.py out.mp4 a.mp4 "label a" b.mp4 "label b" ...
"""

import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 16
HEIGHT = 480


def panel(frame, label):
    image = Image.fromarray(frame)
    image = image.resize((round(image.width * HEIGHT / image.height), HEIGHT), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 22)
    box = draw.textbbox((12, 10), label, font=font)
    draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(0, 0, 0))
    draw.text((12, 10), label, font=font, fill=(255, 255, 255))
    return np.asarray(image)


def main():
    out, pairs = sys.argv[1], sys.argv[2:]
    videos = [(np.stack(imageio.mimread(path, memtest=False)), label) for path, label in zip(pairs[::2], pairs[1::2])]
    frames = min(len(v) for v, _ in videos)
    stacked = [np.concatenate([panel(v[i], label) for v, label in videos], axis=1) for i in range(frames)]
    imageio.mimwrite(out, stacked, fps=FPS, quality=8)
    imageio.imwrite(out.replace(".mp4", ".png"), np.concatenate([stacked[0], stacked[frames // 2], stacked[-1]], axis=0))
    print(f"wrote {out}: {frames} frames, {stacked[0].shape[1]}x{stacked[0].shape[0]}")


if __name__ == "__main__":
    main()
