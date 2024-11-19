# Copyright (c) Facebook, Inc. and its affiliates.

"""
An awesome colormap for really neat visualizations.
Copied from Detectron, and removed gray colors.
"""

import numpy as np

__all__ = ["colormap", "random_color"]

# fmt: off
# RGB:
_COLORS = np.array(
    [
        0.000, 0.000, 1.000,
        1.000, 0.000, 0.000
    ]
).astype(np.float32).reshape(-1, 3)
# fmt: on


def colormap(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a float32 array of Nx3 colors, in range [0, 255] or [0, 1]
    """
    assert maximum in [255, 1], maximum
    c = _COLORS * maximum
    if not rgb:
        c = c[:, ::-1]
    return c


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


if __name__ == "__main__":
    import cv2

    size = 100
    H, W = 10, 10
    canvas = np.random.rand(H * size, W * size, 3).astype("float32")
    for h in range(H):
        for w in range(W):
            idx = h * W + w
            if idx >= len(_COLORS):
                break
            canvas[h * size : (h + 1) * size, w * size : (w + 1) * size] = _COLORS[idx]
    cv2.imshow("a", canvas)
    cv2.waitKey(0)
