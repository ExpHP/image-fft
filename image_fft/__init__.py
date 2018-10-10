#!/usr/bin/env python

from __future__ import print_function, division

import os
import numpy as np
import numpy.fft as npfft
from PIL import Image
try:
    from urllib import urlopen
    try:
        from cStringIO import StringIO as BytesIO
    except ImportError:
        from StringIO import StringIO as BytesIO
except ImportError:
    from urllib.request import urlopen
    from io import BytesIO

try:
    import pyfftw
    use_fftw = True
except ImportError:
    use_fftw = False

def cli_main():
    import argparse
    p = argparse.ArgumentParser(
        description='Compute the DFT of an image.',
        epilog=(
            'Supports a variety of image formats through the Pillow library, '
            'and can transform images of arbitrary dimension thanks to the '
            'unadulterated awesomeness of FFTW3.'
        ),
    )
    p.add_argument(
        'INPUT', type=str,
        help='input path or URL',
    )
    p.add_argument(
        '-o', '--output', type=str,
        help='output file. Default behavior is to generate a name.',
    )

    # TODO: Wacky idea: We could support "complex" images which encode their phase in the hue
    p.add_argument(
        '--channel', type=str,
        choices=[s for (s, _) in LUMINANCE_FUNCS_LIST],
        default='lum-a',
        help=(
            'The input image must be simplified down to a scalar field before performing '
            'the FFT. This lets you chose how that scalar field is determined.'
        ),
    )

    p.add_argument(
        '--mean',
        choices=['cap', 'zero', 'keep'],
        default='cap',
        help=(
            'How to treat the mean image value during normalization of the output image. '
            '--mean=keep does nothing to it, which may be good if you desire reversiblility. '
            '--mean=zero subtracts the mean, enhancing the visiblity of details away from '
            '(0,0) while leaving an unsightly black dot at (0, 0). '
            '--mean=cap produces an identical image to --mean=zero but without the black dot, '
            'by clipping this pixel to the maximum value found elsewhere.'
        ),
    )

    p.add_argument(
        '--no-recenter', dest='recenter', action='store_false',
        help=(
            'Use the output from FFTW as is, which lists points at k = [0, 1, ..., N-1]. '
            'Without this, the default behavior is to recenter each axis around k=0.'
        ),
    )

    p.add_argument(
        '--output', metavar='MODE',
        choices=['real', 'abs-imag', 'norm', 'complex'],
        default='norm',
        help=(
            'Output function. real and imaginary projections are performed after '
            'normalization. Complex encodes the phase in the hue.',
        ),
    )

    p.add_argument(
        '--out-exponent', metavar='EXP',
        default=1.0, type=float,
        help=(
            'Maps amplitude to amplitude**EXP just before producing the output image. '
            'Values greater than 1 will enhance the sharpness of high-intensity regions. '
            'Values less than 1 will enhance regions of moderate intensity.'
        ),
    )

    args = p.parse_args()

    # Get input data
    input_url = to_url(args.INPUT)
    with urlopen(input_url) as f:
        input_image = Image.open(f)

        # images are lazily loaded.
        # Force it to actually read so we can close the file!
        _ = np.array(input_image)

    if args.output is None:
        output_path = default_output_filename(input_url)
    else:
        output_path = args.output

    output_image = compute(
        input_image=input_image,
        channel=args.channel,
        mean_mode=args.mean,
        exponent=args.out_exponent,
        recenter=args.recenter,
        output_mode=args.output,
    )

    output_image.save(output_path)

    # inform user of the default-generated path
    if args.output is None:
        print('Wrote {!r}'.format(output_path))

def compute(
    input_image,
    channel,
    mean_mode,
    exponent,
    recenter,
    output_mode,
):
    scalar_field = get_input_scalar_field(input_image, channel)
    return compute_from_scalar_field(scalar_field, mean_mode, exponent, recenter, output_mode)

def get_input_scalar_field(
    input_image,
    channel,
):
    # White is a safer background color than black in consideration of normalization
    input_image = fill_transparency(input_image, (255, 255, 255))

    pixel_bytes = np.array(input_image.convert('RGB'))
    component_floats = floatify(components_from_pixels(pixel_bytes))
    luminance_floats = LUMINANCE_FUNCS[channel](*component_floats)
    return luminance_floats

def scalar_field_to_image(data):
    from image_fft import np_colorsys

    if issubclass(data.dtype.type, np.floating):
        return Image.fromarray(byteify(data), 'L')
    elif issubclass(data.dtype.type, np.complexfloating):
        norm = np.absolute(data)
        hue = np.angle(data) / (2 * np.pi) % 1.0
        sat = 0.7 * (1-norm) + 1.0 * norm
        val = 0.0 * (1-norm) + 1.0 * norm
        r,g,b = np_colorsys.hsv_to_rgb(hue, sat, val)
        float_data = np.array([r,g,b]).transpose(1, 2, 0) # -> (h, w, 3)
        return Image.fromarray(byteify(float_data), 'RGB')
    else:
        raise TypeError('invalid type for scalar field: {}'.format(data.dtype))

def compute_from_scalar_field(
    scalar_field,
    mean_mode,
    exponent,
    recenter,
    output_mode,
):
    complex_output = do_fft(scalar_field)

    # Post-process
    output_field, output_norm = normalized_output(complex_output, mean_mode=mean_mode, output_mode=output_mode)
    output_field *= output_norm**(exponent - 1)
    if recenter:
        output_field = npfft.fftshift(output_field)

    return scalar_field_to_image(output_field)

def do_fft(scalar_field):
    if use_fftw:
        return _do_fft_pyfftw(scalar_field)
    else:
        return _do_fft_numpy(scalar_field)

def _do_fft_numpy(scalar_field):
    return npfft.fft2(scalar_field)

def _do_fft_pyfftw(scalar_field):
    # Do the thing!
    #
    # FIXME: I'm not entirely sure what's going on here, but if you try to use
    #        the very first FFTW ever instantiated then a large white line will
    #        show up on at least one of my inputs.
    #        To resolve it you must instantiate it twice using different c'pyfftw'opies
    #        of input_array.
    #
    # NOTE: The issue seems to have something to do with wisdom because
    #       calling pyfftw.forget_wisdom() after the first runner is created
    #       causes the issue to return.
    for _ in range(2):
        input_array = np.array(scalar_field, dtype=np.complex128)
        output_array = np.zeros(scalar_field.shape, dtype=np.complex128)
        runner = pyfftw.FFTW(input_array, output_array, axes=(0, 1))

    runner.execute()

    return runner.output_array

def fill_transparency(image, fill_color):
    if image.mode in ('RGBA', 'LA'):
        background = Image.new(image.mode[:-1], image.size, fill_color)
        background.paste(image, image.split()[-1]) # omit transparency
        return background
    else:
        return image

def components_from_pixels(arr):
    return arr.transpose(2, 0, 1)

def pixels_from_components(rgb):
    r, g, b = rgb # check length
    return np.array([r, g, b]).transpose(1, 2, 0)

def byteify(floats):
    assert np.min(floats) > -1e-5, np.min(floats)
    assert np.max(floats) < 1 + 1e-5, np.max(floats)
    floats = clip_array(floats, (0, 1))
    return np.array(np.round(floats * 255), dtype=np.uint8)

def floatify(bytes):
    assert issubclass(bytes.dtype.type, np.integer)
    assert np.min(bytes) >= 0
    assert np.max(bytes) <= 255
    return bytes / 255

def clip_array(array, ivl):
    array = array.copy()
    low, high = ivl
    array[array < low] = low
    array[array > high] = high
    return array

def to_url(path_or_url):
    if '://' in path_or_url:
        return path_or_url
    else:
        return 'file://' + os.path.join(os.getcwd(), path_or_url)

# Default filename is full of DWIM hacks just to make life easier
# for users on windows
def default_output_filename(input_url):
    if input_url.lower().startswith('file://'):
        basename, ext = input_url[len('file://'):].rsplit('.', 1)
        if os.name == 'nt':
            return basename + ' - FFT.' + ext
        else:
            return basename + '-fft.' + ext
    else:
        from datetime import datetime

        # FIXME use proper datetime formatting api
        ts = str(datetime.now())
        for char in ' .:/':
            ts = ts.replace(char, '-')
        if os.name == 'nt':
            return 'FFT Output {}.png'.format(ts)
        else:
            return 'fft-{}.png'.format(ts)

# Wikipedia
def luminance_wiki(r, g, b): return 0.2126*r + 0.7152*g + 0.0722*b
# same as what PIL's L <-> RGB conversion uses
def luminance_pil(r, g, b): return 0.299*r + 0.587*g + 0.114*b
# same coeffs but on squares
def luminance_pil2(r, g, b): return (0.299*r*r + 0.587*g*g + 0.114*b*b)**0.5

# using a list for fixed order
LUMINANCE_FUNCS_LIST = [
    ('red', (lambda r, _g, _b: r)),
    ('green', (lambda _r, g, _b: g)),
    ('blue', (lambda _r, _g, b: b)),
    ('lum-a', luminance_wiki),
    ('lum-b', luminance_pil),
    ('lum-c', luminance_pil2),
]
LUMINANCE_FUNCS = dict(LUMINANCE_FUNCS_LIST)

# FIXME this will produce NaN on a 1x1 image (like anyone cares)
def normalized_output(complex_array, mean_mode, output_mode):
    norm = np.absolute(complex_array)

    out = {
        'abs-real': (lambda: complex_array.real),
        'abs-imag': (lambda: np.absolute(complex_array.imag)),
        'norm': (lambda: norm.copy()),
        'complex': (lambda: complex_array.copy()),
    }[output_mode]()

    if mean_mode == 'cap':
        # maximum absolute value without mean
        norm_mean = norm[0, 0]
        norm[0, 0] = 0
        # not a naming inconsistency; norm_mean is the norm of the (complex) mean.
        # max_norm is the max of the norms!
        max_norm = np.max(norm)
        # put mean back in, clipped.
        norm[0, 0] = min(norm_mean, max_norm)
        out[0, 0] *= 1 if norm_mean == 0 else min(norm_mean, max_norm)/norm_mean

    elif mean_mode == 'zero':
        norm[0, 0] = 0
        out[0, 0] = 0
        max_norm = np.max(norm)

    elif mean_mode == 'keep':
        max_norm = np.max(norm)

    else:
        assert False, 'complete switch'

    norm /= 1 if max_norm == 0 else max_norm
    out /= 1 if max_norm == 0 else max_norm
    return out, norm
