import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from api.app.main import _runtime_kana_image_svg

try:
    print("Testing Katakana conversion...")
    svg_kata = _runtime_kana_image_svg("カタカナ", "watermark")
    print("Katakana SVG Length:", len(svg_kata))
    print("Katakana has <path>?", "<path" in svg_kata)
    
    print("\nTesting Kanji conversion...")
    svg_kanji = _runtime_kana_image_svg("漢字", "watermark")
    print("Kanji SVG Length:", len(svg_kanji))
    print("Kanji has <path>?", "<path" in svg_kanji)
    
except Exception as e:
    print("Exception:", e)
