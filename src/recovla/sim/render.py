"""記録用カメラの描画（流用元 teleop/collect.py の close_renderer・render_cameras、処理は変えていない）。"""
import mujoco

from recovla.sim.control import CAMERAS


def close_renderer(renderer: mujoco.Renderer) -> None:
    """Close a mujoco.Renderer without corrupting the others.

    mujoco 3.2.3's Renderer.close() (also run from __del__) frees its GL
    context first and its MjrContext second, so the MjrContext's GL objects
    are deleted in whichever context is current -- typically another live
    renderer's. Measured 2026-09-16: closing one renderer after the
    recording renderer had drawn changed every pixel of the next recorded
    frames. Here the renderer's own context is made current first. Never
    let a Renderer be garbage-collected while another one is still in use;
    close it with this instead.
    """
    gl = renderer._gl_context
    if gl:
        gl.make_current()
    if renderer._mjr_context:
        renderer._mjr_context.free()
    renderer._mjr_context = None
    if gl:
        gl.free()
    renderer._gl_context = None


def render_cameras(renderer: mujoco.Renderer, data, cameras=CAMERAS) -> list:
    frames = []
    for name in cameras:
        renderer.update_scene(data, camera=name)
        frames.append(renderer.render().copy())
    return frames
