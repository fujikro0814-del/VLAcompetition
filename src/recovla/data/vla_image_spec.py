"""The ONE place that defines how recorded camera images become policy images.

Raw episodes store the renders exactly as the cameras see them (03_収録/raw, never transformed).
Every image that enters a policy -- the LeRobot dataset written by convert_to_lerobot.py and the
observation built at evaluation time by vla_observation.py -- goes through to_policy_image() /
policy_images() here. Do not flip or rotate policy images anywhere else: a transform written in
one path and forgotten in the other fails silently.

Net transforms (raw render -> policy image), fixed 2026-09-17:

  overhead (O1)   fliplr   LIBERO overhead frames are upright but mirrored (Step C 5.3)
  wrist    (W6m)  flipud   with it, W6m's image motion matches lerobot/smolvla_libero's wrist
                           images: 54/54 flow matrices, no other net transform matched any
                           (wrist-camera remount Step E-A)

Only the four net names below are allowed; never compose them (fliplr + rot180 == flipud).
Depends on numpy only, so both the teleop Python (python311) and the LeRobot venv import it.
"""
import numpy as np

SPEC_VERSION = 2              # 1 = converter v1: both views flipped left-right (before Step E)
CAMERAS = ("overhead", "wrist")   # must equal teleop.collect.CAMERAS (checked by a test)
NET_TRANSFORMS = ("identity", "fliplr", "flipud", "rot180")
VIEW_TRANSFORMS = {"overhead": "fliplr", "wrist": "flipud"}
IMAGE_KEYS = {"overhead": "observation.images.image", "wrist": "observation.images.image2"}

_APPLY = {
    "identity": lambda a: a,
    "fliplr": lambda a: a[:, ::-1],
    "flipud": lambda a: a[::-1],
    "rot180": lambda a: a[::-1, ::-1],
}


class ImageSpecError(ValueError):
    """Raised whenever views, transform names or a recorded spec do not match this module."""


def validate(view_transforms=None, image_keys=None, cameras=None) -> None:
    """Raise ImageSpecError unless the three tables name exactly the same views and every
    transform is a known net transform. Called at import time on the module's own tables."""
    vt = VIEW_TRANSFORMS if view_transforms is None else view_transforms
    ik = IMAGE_KEYS if image_keys is None else image_keys
    cams = CAMERAS if cameras is None else cameras
    if set(vt) != set(cams) or set(ik) != set(cams) or len(set(cams)) != len(cams):
        raise ImageSpecError(f"views differ: transforms {sorted(vt)}, image keys {sorted(ik)}, cameras {list(cams)}")
    bad = {v: t for v, t in vt.items() if t not in NET_TRANSFORMS}
    if bad:
        raise ImageSpecError(f"unknown net transform(s) {bad}; allowed {NET_TRANSFORMS}")


validate()


def apply_net_transform(name: str, raw_rgb: np.ndarray) -> np.ndarray:
    """Apply one named net transform (for checks that compare against the alternatives).
    Policy images must use to_policy_image / policy_images, which pick the transform by view."""
    if name not in NET_TRANSFORMS:
        raise ImageSpecError(f"unknown net transform {name!r}; allowed {NET_TRANSFORMS}")
    a = np.asarray(raw_rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ImageSpecError(f"expected H x W x 3, got shape {a.shape}")
    return np.ascontiguousarray(_APPLY[name](a)).copy()


def to_policy_image(view: str, raw_rgb: np.ndarray) -> np.ndarray:
    """Raw H x W x 3 image of one camera -> the image the policy sees (C-contiguous copy)."""
    if view not in VIEW_TRANSFORMS:
        raise ImageSpecError(f"unknown view {view!r}; known {sorted(VIEW_TRANSFORMS)}")
    return apply_net_transform(VIEW_TRANSFORMS[view], raw_rgb)


def policy_images(raw_by_view: dict) -> dict:
    """{view: raw image} -> {dataset feature key: policy image}. The views must be exactly the
    spec's views: a missing or an extra view raises instead of silently dropping a camera."""
    got, want = set(raw_by_view), set(VIEW_TRANSFORMS)
    if got != want:
        raise ImageSpecError(f"views {sorted(got)} != spec views {sorted(want)} "
                             f"(missing {sorted(want - got)}, extra {sorted(got - want)})")
    return {IMAGE_KEYS[v]: to_policy_image(v, raw_by_view[v]) for v in CAMERAS}


def spec_record() -> dict:
    """What convert_to_lerobot.py writes into meta/conversion.json."""
    return {"image_spec_version": SPEC_VERSION, "image_transforms": dict(VIEW_TRANSFORMS),
            "image_keys": dict(IMAGE_KEYS)}


def check_record(conversion: dict, source: str = "conversion.json") -> None:
    """Raise ImageSpecError unless a conversion.json (dict) was written with exactly this spec.
    Version-1 datasets (no image_transforms; both views flipped left-right) are rejected."""
    version = conversion.get("image_spec_version")
    if version != SPEC_VERSION:
        raise ImageSpecError(f"{source}: image_spec_version {version!r} != {SPEC_VERSION} "
                             f"(converter_version {conversion.get('converter_version')!r}); re-convert the "
                             f"raw episodes with the current convert_to_lerobot.py")
    for field, want in (("image_transforms", VIEW_TRANSFORMS), ("image_keys", IMAGE_KEYS)):
        if conversion.get(field) != want:
            raise ImageSpecError(f"{source}: {field} {conversion.get(field)!r} != runtime {want!r}")
