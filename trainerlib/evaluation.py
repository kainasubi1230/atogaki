def similarity_score(reference_count: int, generated_count: int) -> float:
    if reference_count <= 0 and generated_count <= 0:
        return 1.0
    max_count = max(reference_count, generated_count, 1)
    diff = abs(reference_count - generated_count)
    score = 1.0 - (diff / max_count)
    return round(max(0.0, min(1.0, score)), 3)


def cer_estimate(expected: str, generated: str) -> float:
    if not expected:
        return 0.0
    mismatches = 0
    for idx, ch in enumerate(expected):
        if idx >= len(generated) or generated[idx] != ch:
            mismatches += 1
    mismatches += max(0, len(generated) - len(expected))
    return round(mismatches / max(len(expected), 1), 3)

