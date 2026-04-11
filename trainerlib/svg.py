def trajectory_to_svg(trajectory: list[dict], watermark_text: str) -> str:
    if not trajectory:
        return (
            "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='160'>"
            f"<text x='10' y='140' fill='#d11'>{watermark_text}</text></svg>"
        )

    width = max(p["x"] for p in trajectory) + 40
    height = max(p["y"] for p in trajectory) + 60

    path_parts: list[str] = []
    pen_down = False
    for p in trajectory:
        cmd = "M" if not pen_down or p["pen_state"] != "down" else "L"
        path_parts.append(f"{cmd}{p['x']},{p['y']}")
        pen_down = p["pen_state"] == "down"

    path_data = " ".join(path_parts)
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<path d='{path_data}' stroke='black' fill='none' stroke-width='2'/>"
        f"<text x='12' y='{height - 14}' fill='#c62828' font-size='14'>{watermark_text}</text>"
        "</svg>"
    )

