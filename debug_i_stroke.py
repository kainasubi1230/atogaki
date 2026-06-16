#!/usr/bin/env python3
"""Debug script to check 「い」 stroke width in trajectory_to_svg()"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from trainerlib.svg import trajectory_to_svg
from api.app.main import _generate_hiragana_runtime_trajectory

# Generate trajectory for 「い」
print("=== Generating trajectory for 「い」 ===")
trajectory = _generate_hiragana_runtime_trajectory("い")

if not trajectory:
    print("ERROR: No trajectory generated!")
    sys.exit(1)

print(f"Trajectory length: {len(trajectory)} points")
print("\nFirst 10 points:")
for i, point in enumerate(trajectory[:10]):
    print(f"  {i}: x={point.get('x')}, y={point.get('y')}, pen_state={point.get('pen_state')}, width={point.get('width')}")

# Check width statistics
widths = [p.get('width', 1) for p in trajectory if p.get('pen_state') == 'down']
if widths:
    print(f"\nWidth statistics (down strokes only):")
    print(f"  Min: {min(widths)}")
    print(f"  Max: {max(widths)}")
    print(f"  Avg: {sum(widths) / len(widths):.2f}")
    print(f"  Count: {len(widths)}")

# Generate SVG
print("\n=== Generating SVG ===")
svg = trajectory_to_svg(trajectory, "テスト")
print(f"SVG length: {len(svg)} chars")

# Extract stroke-width values from SVG
import re
stroke_widths = re.findall(r"stroke-width='([\d.]+)'", svg)
if stroke_widths:
    print(f"\nStroke-width values in SVG:")
    print(f"  Values: {stroke_widths}")
    widths_float = [float(w) for w in stroke_widths]
    print(f"  Min: {min(widths_float):.3f}")
    print(f"  Max: {max(widths_float):.3f}")
    print(f"  Avg: {sum(widths_float) / len(widths_float):.3f}")

# Save SVG for inspection
output_path = Path(__file__).parent / "debug_i_output.svg"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"\nSVG saved to: {output_path}")
