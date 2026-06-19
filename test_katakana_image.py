import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from trainerlib.kana_image import generate_kana_image
from api.app.main import _strokes_from_runtime_image, _build_svg_string

try:
    print("Generating Katakana image...")
    img = generate_kana_image("カ", style_seed="test", size=160)
    print("Image generated:", img.size)

    strokes = _strokes_from_runtime_image(img)
    print("Strokes extracted:", len(strokes))
    if not strokes:
        print("EMPTY STROKES! This is why it fails!")
    
    # Try formatting as SVG
    stroke_paths = [(strokes, 0.0, 0.0, 1.0, 160.0, False)]
    svg = _build_svg_string(stroke_paths, 160, 160, "watermark")
    print("SVG LENGTH:", len(svg))
    print(svg[:500])
except Exception as e:
    print("Exception:", e)
