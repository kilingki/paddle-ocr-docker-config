from dataclasses import dataclass
from io import BytesIO
import base64, hashlib, warnings
from PIL import Image, UnidentifiedImageError
from .errors import APIError


@dataclass(frozen=True)
class PreparedImage:
    sha256: str
    width: int
    height: int
    format: str
    data_url: str

    def metadata(self):
        return {"sha256": self.sha256, "width": self.width,
                "height": self.height, "format": self.format}


def prepare_image(data, settings):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in ("PNG", "JPEG"):
                    raise APIError(415, "UNSUPPORTED_MEDIA_TYPE", "only PNG and JPEG are supported")
                width, height = image.size
                if (min(width, height) <= 0 or max(width, height) > settings.max_image_edge
                        or width * height > settings.max_image_pixels
                        or max(width, height) / min(width, height) > 200):
                    raise APIError(400, "BAD_REQUEST", "image dimensions exceed supported limits")
                if getattr(image, "n_frames", 1) != 1:
                    raise APIError(400, "BAD_REQUEST", "multi-frame images are not supported")
                fmt = image.format
                image.load()
                # No EXIF rotation, crop, scaling, deskew, or semantic processing.
                rgb = image.convert("RGB")
                out = BytesIO()
                rgb.save(out, format="PNG")
        return PreparedImage(hashlib.sha256(data).hexdigest(), width, height, fmt,
                             "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii"))
    except APIError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise APIError(400, "BAD_REQUEST", "invalid or damaged image") from exc
