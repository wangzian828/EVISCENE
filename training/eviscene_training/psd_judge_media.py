"""Lossless transport adaptation; original task/archive identities never change."""
import base64
import hashlib
import io

from PIL import Image


def judge_image_transport(blob):
    """Keep accepted bytes verbatim; encode single-frame AVIF pixels as PNG.

    Gemini's documented inline formats include PNG, not AVIF. Do not rewrite
    source files, resize, flatten alpha, drop animation or silently use JPEG.
    The conversion record is included in the judge's bound material packet.
    """
    conversion = None
    with Image.open(io.BytesIO(blob)) as original:
        mime = Image.MIME.get(original.format)
        if original.format == 'AVIF':
            if getattr(original, 'n_frames', 1) != 1 or original.mode not in ('RGB', 'RGBA'):
                raise ValueError('PSD judge cannot losslessly transport this AVIF layout')
            original.load()
            pixels = original.tobytes()
            metadata = {k: original.info[k] for k in ('icc_profile', 'exif') if original.info.get(k)}
            output = io.BytesIO()
            original.save(output, format='PNG', **metadata)
            wire = output.getvalue()
            with Image.open(io.BytesIO(wire)) as restored:
                restored.load()
                if (restored.mode != original.mode or restored.size != original.size
                        or restored.tobytes() != pixels
                        or dict(restored.getexif()) != dict(original.getexif())
                        or restored.info.get('icc_profile') != original.info.get('icc_profile')):
                    raise ValueError('PSD judge PNG transport changed decoded AVIF content')
            conversion = {'version': 'ifv-psd-avif-png-transport-v1',
                'source_sha256': hashlib.sha256(blob).hexdigest(), 'source_mime': mime,
                'transport_sha256': hashlib.sha256(wire).hexdigest(), 'transport_mime': 'image/png',
                'decoded_pixels_sha256': hashlib.sha256(pixels).hexdigest(),
                'mode': original.mode, 'size': list(original.size), 'frames': 1}
            blob, mime = wire, 'image/png'
        elif mime in {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}:
            original.verify()
        else:
            raise ValueError('unsupported PSD judge image type')
    return {'type': 'image', 'mime_type': mime, 'data': base64.b64encode(blob).decode()}, conversion
