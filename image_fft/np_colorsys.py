import numpy as np

def rgb_to_hsv(r, g, b):
    # FIXME proper broadcasting
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    if len(set([r.shape, g.shape, b.shape])) > 1:
        raise ValueError('r, g, b must all have same shape!')

    maxc = r.copy()
    maxc = np.where(g > maxc, g, maxc)
    maxc = np.where(b > maxc, b, maxc)
    minc = r.copy()
    minc = np.where(g < minc, g, minc)
    minc = np.where(b < minc, b, minc)
    if np.min(minc) < 0.0 or np.max(maxc) > 1.0:
        raise ValueError('r, g, b must lie in [0, 1]')

    h = np.zeros(r.shape)
    s = np.zeros(r.shape)
    v = maxc

    # Mask away areas of degenerate hue, leaving them at 0 hue/saturation
    mask = minc != maxc

    diffc = maxc - minc
    s[mask] = diffc[mask] / maxc[mask]
    rc = (maxc - r)[mask] / diffc[mask]
    gc = (maxc - g)[mask] / diffc[mask]
    bc = (maxc - b)[mask] / diffc[mask]

    h[mask] = np.where((b == maxc)[mask], 4.0 + gc - rc, h[mask])
    h[mask] = np.where((g == maxc)[mask], 2.0 + rc - bc, h[mask])
    h[mask] = np.where((r == maxc)[mask], 0.0 + bc - gc, h[mask])
    h = h/6.0 % 1.0
    return h, s, v

def hsv_to_rgb(h, s, v):
    # FIXME proper broadcasting
    h = np.array(h)
    s = np.array(s)
    v = np.array(v)
    if len(set([h.shape, s.shape, v.shape])) > 1:
        raise ValueError('h, s, v must all have same shape!')

    # if s == 0.0:
    #     return v, v, v
    i = np.floor(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6.0

    r = np.zeros(h.shape)
    g = np.zeros(h.shape)
    b = np.zeros(h.shape)
    for si, (sr, sg, sb) in enumerate([
        (v, t, p), (q, v, p),
        (p, v, t), (p, q, v),
        (t, p, v), (v, p, q),
    ]):
        mask = i == si
        r[mask] = sr[mask]
        g[mask] = sg[mask]
        b[mask] = sb[mask]
    return r, g, b

# Unit tests
if __name__ == '__main__':
    import colorsys
    import time
    from numpy import testing

    floats = [-0.0] + [i / 255 for i in range(256)]
    N = len(floats)

    # arrays constant along all but one axis
    x1 = np.array([[floats] * N] * N)
    x2 = np.array([[[x] * N for x in floats]] * N)
    x3 = np.array([[[x] * N for _ in floats] for x in floats])

    print('verifying {} x {} x {} = {} inputs against colorsys.rgb_to_hsv'.format(N, N, N, N**3))
    t0 = time.time()
    expect_h, expect_s, expect_v = np.vectorize(colorsys.rgb_to_hsv)(x1, x2, x3)
    print('  - time spent in colorsys: {:.3}s'.format(time.time() - t0))

    t0 = time.time()
    actual_h, actual_s, actual_v = rgb_to_hsv(x1, x2, x3)
    print('  - time spent in our code: {:.3}s'.format(time.time() - t0))

    testing.assert_array_equal(actual_h, expect_h)
    testing.assert_array_equal(actual_s, expect_s)
    testing.assert_array_equal(actual_v, expect_v)

    print('verifying {} x {} x {} = {} inputs against colorsys.hsv_to_rgb'.format(N, N, N, N**3))
    t0 = time.time()
    expect_r, expect_g, expect_b = np.vectorize(colorsys.hsv_to_rgb)(x1, x2, x3)
    print('  - time spent in colorsys: {:.3}s'.format(time.time() - t0))

    t0 = time.time()
    actual_r, actual_g, actual_b = hsv_to_rgb(x1, x2, x3)
    print('  - time spent in our code: {:.3}s'.format(time.time() - t0))

    testing.assert_array_equal(actual_r, expect_r)
    testing.assert_array_equal(actual_g, expect_g)
    testing.assert_array_equal(actual_b, expect_b)
