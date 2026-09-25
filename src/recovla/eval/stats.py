"""流用元 gui/gui_core.py の wilson_interval() だけを切り出したもの（B_提案書 §2.2）。"""
import math


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    """95 % interval for a proportion (the interval used in the reports)."""
    if trials <= 0:
        return None, None
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
