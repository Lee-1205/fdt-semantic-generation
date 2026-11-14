from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "plots" / "legacy_vs_current_entropy.png"
VALUES = [
    ("Early cdist", 0.9993, "#777777"),
    ("Current FDT-RLM", 0.9679, "#2878b5"),
    ("Revised SDFT epoch 19", 0.7401, "#55a868"),
    ("Dual-anchor FDT2 epoch 4", 0.3608, "#c44e52"),
]


def font(size: int, bold: bool = False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


canvas = Image.new("RGB", (1200, 650), "white")
draw = ImageDraw.Draw(canvas)
draw.text((70, 38), "Legacy vs current routing entropy", fill="#222222", font=font(34, True))
draw.text((70, 88), "Normalized selected-membership entropy H / ln(Top-K)", fill="#555555", font=font(22))

axis_left = 350
axis_right = 1110
axis_top = 150
axis_bottom = 555
for tick in range(6):
    value = tick / 5
    x = axis_left + int((axis_right - axis_left) * value)
    draw.line((x, axis_top, x, axis_bottom), fill="#dddddd", width=2)
    draw.text((x - 15, axis_bottom + 12), f"{value:.1f}", fill="#555555", font=font(18))

bar_height = 58
for index, (label, value, color) in enumerate(VALUES):
    y = axis_top + 28 + index * 93
    draw.text((70, y + 13), label, fill="#222222", font=font(22))
    width = int((axis_right - axis_left) * value)
    draw.rectangle((axis_left, y, axis_left + width, y + bar_height), fill=color)
    draw.text((axis_left + width + 14, y + 13), f"{value:.3f}", fill="#222222", font=font(22, True))

draw.line((axis_right, axis_top, axis_right, axis_bottom), fill="#333333", width=3)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT)
print(OUTPUT)
