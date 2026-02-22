#!/bin/python

import sys
_np = None

def spiral(N, initial_direction = None, center = None, use_numpy = False):
    global _np
    if use_numpy:
        _validate_sprial(N, initial_direction, center, use_numpy)
        if _np is None:
            import numpy as np
            _np = np

        i, j = _np.indices((N, N))
        return _spiral_impl_numpy(i, j, N)

    center, parsed_initial_direction = _validate_sprial(N, initial_direction, center, use_numpy)
    return _spiral_impl_plain(N, parsed_initial_direction, center)


def diagonal_sum(matrix, use_numpy = False):
    global _np
    if use_numpy:
        if _np is None:
            import numpy as np
            _np = np
        return _diagonal_sum_impl_numpy(_validate_diagonal_sum(matrix, use_numpy))

    _validate_diagonal_sum(matrix, use_numpy)
    return _diagonal_sum_impl_plain(matrix)

def _die(msg):
    sys.stderr.write(f"{msg} \n")
    exit(1)

def _validate_diagonal_sum(matrix, use_numpy):
    if use_numpy:
        try:
            arr = _np.asarray(matrix)
        except Exception as e:
            _die(f"failed to convert to numpy array: {matrix!r}\nerror: {e}")

        if arr.ndim != 2:
            _die(f"not a 2D matrix (ndim={arr.ndim})")

        n_rows, n_cols = arr.shape
        if n_rows != n_cols:
            _die(f"not a square matrix (shape={arr.shape})")

        return arr

    if not isinstance(matrix, list):
        _die("not a matrix (expected list of rows)")

    N = len(matrix)
    if N == 0:
        _die("not a square matrix (empty)")

    if not all(isinstance(row, list) for row in matrix):
        _die("not a matrix (each row must be a list)")

    if not all(len(row) == N for row in matrix):
        _die("not a square matrix (rows must all have length N)")


def _validate_sprial(N, initial_direction, center, use_numpy):

    if not isinstance(N, int) or N <= 0:
        _die(f"invalid N: expected positive integer, got {N!r}")

    if use_numpy:
        if N % 2 == 0:
            _die(f"invalid N: numpy only supports odd numbers (N={N})")
        if center is not None or initial_direction is not None:
            _die(f"cannot specify center or initial_direction in numpy implementation; \
                 (center={center}, initial_direction={initial_direction}) ")
        return

    if N % 2 == 1 and center is not None:
        _die(f"invalid center: N is odd (N={N})")

    if initial_direction is not None and center is not None:
        _die(
            f"initial_direction and center cannot both be specified "
            f"(initial_direction={initial_direction!r}, center={center!r})"
        )

    # Defaults
    if initial_direction is None and center is None:
        initial_direction = "right"
        center = "top-left"

    center_to_dir = {
        "top-left": "right",
        "top-right": "down",
        "bottom-left": "up",
        "bottom-right": "left",
    }
    dir_to_center = {
        "right": "top-left",
        "down": "top-right",
        "left": "bottom-right",
        "up": "bottom-left",
    }
    dir_to_step = {
        "right": (0, 1),
        "left": (0, -1),
        "down": (1, 0),
        "up": (-1, 0),
    }

    if N % 2 == 0:
        if center is not None:
            try:
                initial_direction = center_to_dir[center]
            except KeyError:
                _die(f"invalid center: {center!r}")
        else:
            # initial_direction is not None here (due to defaults / mutual exclusion)
            try:
                center = dir_to_center[initial_direction]
            except KeyError:
                _die(f"invalid initial_direction: {initial_direction!r}")

    try:
        direction = dir_to_step[initial_direction]
    except KeyError:
        # Unreachable
        _die(f"invalid initial_direction: {initial_direction!r}")

    return center, direction


def _spiral_impl_plain(N, initial_direction, center):
    matrix = [[0] * N for _ in range(N)]
    number = 1
    repetition = 0

    x = y = (N - 1) // 2
    if center == "bottom-left" or center == "bottom-right":
        x += 1
    if center == "top-right" or center == "bottom-right":
        y += 1

    matrix[x][y] = number # Fill the center

    direction = initial_direction
    done = False
    while not done:
        repetition += 1

        for _ in range(2):
            dx, dy = direction
            for _ in range(repetition):
                number += 1
                if (number > N * N):
                    done = True
                    break
                x += dx
                y += dy
                matrix[x][y] = number

            direction = (dy, -dx) # Clockwise turn

    return matrix

def _spiral_impl_numpy(i, j, N):
    center = N // 2
    x = center - i
    y = j - center

    layer = _np.maximum(_np.abs(x), _np.abs(y))
    max_value = (2 * layer + 1) ** 2

    out = _np.empty((N, N), dtype=_np.int64)

    # Exclude with `& ~` for corners
    right  = (x == layer)
    bottom = (y == -layer) & ~right
    left   = (x == -layer) & ~(right | bottom)
    top    = (y == layer) & ~(right | bottom | left)

    out[right]  = max_value[right]  -   (layer[right] - y[right])
    out[bottom] = max_value[bottom] - 2*layer[bottom] - (layer[bottom] - x[bottom])
    out[left]   = max_value[left]   - 4*layer[left]   - (y[left] + layer[left])
    out[top]    = max_value[top]    - 6*layer[top]    - (x[top] + layer[top])

    return out

def _diagonal_sum_impl_plain(matrix):
    N = len(matrix)
    primary_sum = secondary_sum = 0

    for i in range(N):
        primary_sum += matrix[i][i]
        secondary_sum += matrix[i][N - i - 1] # Minus 1 because array starts at 0 not 1

    return primary_sum, secondary_sum

def _diagonal_sum_impl_numpy(matrix):
    return _np.trace(matrix), _np.trace(_np.fliplr(matrix))
