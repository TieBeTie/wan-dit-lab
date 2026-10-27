"""Assemble paired playback and a readable report from the fixed PC experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import html
import json

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/pc_fifo_256p.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    width, height = config["width"], config["height"]
    out = ROOT / config["output"]
    result = json.loads((out / "_data/results.json").read_text(encoding="utf-8"))
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    entries = []
    for pair, metric in zip(config["pairs"], result["pairs"]):
        folder = out / pair["id"]
        comparison = folder / "FIFO_vs_PC.mp4"
        if not comparison.exists():
            left = imageio.get_reader(folder / "FIFO.mp4")
            right = imageio.get_reader(folder / "PC.mp4")
            writer = imageio.get_writer(comparison, fps=16, quality=8)
            banner = Image.new("RGB", (2 * width, 64), "#151922")
            draw = ImageDraw.Draw(banner)
            draw.text((12, 6), "FIFO", font=font, fill="white")
            draw.text((width + 12, 6), "PC · Participative Compression", font=font, fill="white")
            draw.text(
                (12, 34),
                f"{pair['id']} | benchmark line {pair['source_line']} | seed {pair['seed']} | BF16 {height}x{width} | sink 10",
                font=small,
                fill="#cbd5e1",
            )
            banner = np.asarray(banner)
            count = 0
            for a, b in zip(left, right):
                writer.append_data(np.concatenate([banner, np.concatenate([a, b], axis=1)], axis=0))
                count += 1
            writer.close()
            left.close()
            right.close()
            assert count == 321
        entries.append(f'''<article><h2>{pair["id"]} · ΔMUSIQ = {metric["delta"]:+.3f}</h2>
<p>{html.escape(pair["prompt"])}</p><video controls preload="none" src="{pair["id"]}/FIFO_vs_PC.mp4"></video>
<p>FIFO: {metric["FIFO"]["late_musiq"]:.3f} · PC: {metric["PC"]["late_musiq"]:.3f}</p></article>''')
    verdict = {
        "do_not_reject_H0": "H₀ не отвергаем: статистически различимый перевес по выбранному критерию не получен.",
        "reject_H0_PC_higher_MUSIQ": "H₀ отвергаем: в этой серии PC чаще даёт более высокий MUSIQ последних пяти секунд.",
        "reject_H0_PC_lower_MUSIQ": "H₀ отвергаем: в этой серии PC чаще даёт более низкий MUSIQ последних пяти секунд.",
    }[result["decision"]]
    page = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>PC против FIFO</title>
<style>body{{background:#11151d;color:#e7edf5;font:18px/1.5 Arial;margin:40px auto;max-width:1050px;padding:0 20px}}h1,h2{{font-weight:500}}article{{padding:25px 0;border-top:1px solid #394150}}video{{width:100%}}a{{color:#93c5fd}}p{{max-width:1000px}}</style>
<h1>PC против FIFO · {len(config['pairs'])} пар · BF16 {height}×{width}</h1><p>{verdict}</p>
<p>Победы PC / FIFO / ничьи: {result["wins"]} / {result["losses"]} / {result["ties"]}. Двусторонний тест знаков: p = {result["exact_sign_test_p_two_sided"]:.6g}.</p>
<p>Среднее ΔMUSIQ: {result["mean_delta"]:+.3f}; bootstrap 95% интервал среднего: [{result["bootstrap_mean_delta_95ci"][0]:+.3f}, {result["bootstrap_mean_delta_95ci"][1]:+.3f}].</p>
<p>Одинаковые промпт, seed, шум, якорь 10, окно 21, Temporal RoPE, 4 шага, shift 2.67946, TAEHV. PC защищает последний готовый кадр. По 20.0625 секунды на ролик.</p>
<p>MUSIQ оценивает отдельные изображения. Это не проверка соответствия промпту, движения или общего превосходства метода. Кадры внутри ролика не считаются независимыми наблюдениями.</p>
{"".join(entries)}</html>"""
    (out / "Сравнения.html").write_text(page, encoding="utf-8")
    note = ROOT / config.get("note", "notes/research/PC эксперимент.md")
    lines = [
        "# PC против FIFO",
        "",
        verdict,
        "",
        "$$",
        r"\underbrace{N=16}_{\text{пар роликов}},\quad"
        + rf"\underbrace{{{result['wins']}:{result['losses']}}}_{{\text{{победы PC : FIFO}}}},\quad p={result['exact_sign_test_p_two_sided']:.6g}",
        "$$",
        "",
        "$$",
        rf"\underbrace{{\overline{{\Delta Q}}={result['mean_delta']:+.3f}}}_{{\text{{MUSIQ последних 5 с, PC минус FIFO}}}}",
        "$$",
        "",
        f"[Ролики и все промпты]({(out / 'Сравнения.html').as_uri()}) · [Конфиг]({args.config.resolve().as_uri()})",
        "",
        "Один prompt/seed образует одну пару. Основной критерий и двусторонний тест знаков зафиксированы до генерации; α = 0.05. Интервал среднего в HTML описательный, bootstrap по парам. MUSIQ не оценивает движение или соответствие промпту.",
        "",
        "| Пара | FIFO MUSIQ | PC MUSIQ | PC − FIFO |",
        "| :--- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r['id']} | {r['FIFO']['late_musiq']:.3f} | {r['PC']['late_musiq']:.3f} | {r['delta']:+.3f} |"
        for r in result["pairs"]
    ]
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(verdict, flush=True)
    print("Report:", out / "Сравнения.html", flush=True)


if __name__ == "__main__":
    main()
