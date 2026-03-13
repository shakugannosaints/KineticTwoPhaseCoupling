from __future__ import annotations

import math
from dataclasses import dataclass

import glfw
import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FALSE,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINE_SMOOTH,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POSITION,
    GL_PROJECTION,
    GL_SRC_ALPHA,
    GL_TRUE,
    GL_TRIANGLES,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColor4f,
    glDepthMask,
    glDisable,
    glEnable,
    glEnd,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glNormal3f,
    glVertex3f,
    glViewport,
)
from OpenGL.GLU import gluLookAt, gluPerspective


@dataclass
class OrbitCamera:
    yaw: float
    pitch: float
    distance: float
    target: np.ndarray

    def position(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        return self.target + np.array([self.distance * cp * cy, self.distance * sp, self.distance * cp * sy], dtype=np.float64)


class OpenGLViewer:
    def __init__(self, extent: tuple[float, float, float]):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")
        self.window = glfw.create_window(1400, 900, "LD23 Simplified Reproduction", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create OpenGL window.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.camera = OrbitCamera(yaw=-0.82, pitch=0.35, distance=max(extent) * 1.65, target=np.asarray(extent, dtype=np.float64) * np.array([0.5, 0.35, 0.5], dtype=np.float64))
        self.last_cursor = None
        self.left_drag = False
        self.should_reset = False
        self.paused = False
        self.wireframe = None
        self.water_vertices = np.empty((0, 3), dtype=np.float32)
        self.water_faces = np.empty((0, 3), dtype=np.int32)
        self.water_normals = np.empty((0, 3), dtype=np.float32)
        self.dice_vertices = np.empty((0, 3), dtype=np.float32)
        self.dice_faces = np.empty((0, 3), dtype=np.int32)

        glfw.set_window_user_pointer(self.window, self)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LINE_SMOOTH)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.configure_scene("LD23 Simplified Reproduction", extent, (0.5, 0.35, 0.5), 1.65)

    def configure_scene(
        self,
        title: str,
        extent: tuple[float, float, float],
        target_factor: tuple[float, float, float],
        distance_factor: float,
    ) -> None:
        if self.window is None:
            return
        glfw.set_window_title(self.window, title)
        self.camera = OrbitCamera(
            yaw=-0.82,
            pitch=0.35,
            distance=max(extent) * distance_factor,
            target=np.asarray(extent, dtype=np.float64) * np.asarray(target_factor, dtype=np.float64),
        )
        self.last_cursor = None

    def close(self) -> None:
        if self.window is None:
            return
        glfw.set_cursor_pos_callback(self.window, None)
        glfw.set_mouse_button_callback(self.window, None)
        glfw.set_scroll_callback(self.window, None)
        glfw.make_context_current(None)
        glfw.destroy_window(self.window)
        self.window = None
        glfw.terminate()

    def should_close(self) -> bool:
        if self.window is None:
            return True
        return glfw.window_should_close(self.window)

    def poll(self) -> None:
        glfw.poll_events()
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        if glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS:
            self.should_reset = True

    def consume_reset(self) -> bool:
        value = self.should_reset
        self.should_reset = False
        return value

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def set_meshes(
        self,
        wireframe: np.ndarray,
        water_vertices: np.ndarray,
        water_faces: np.ndarray,
        water_normals: np.ndarray,
        dice_vertices: np.ndarray,
        dice_faces: np.ndarray,
    ) -> None:
        self.wireframe = wireframe
        self.water_vertices = water_vertices
        self.water_faces = water_faces
        self.water_normals = water_normals
        self.dice_vertices = dice_vertices
        self.dice_faces = dice_faces

    def render(self) -> None:
        if self.window is None:
            return
        width, height = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, width, max(height, 1))
        glClearColor(0.055, 0.065, 0.09, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / max(height, 1), 0.1, 5000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        eye = self.camera.position()
        target = self.camera.target
        gluLookAt(eye[0], eye[1], eye[2], target[0], target[1], target[2], 0.0, 1.0, 0.0)
        glLightfv(GL_LIGHT0, GL_POSITION, (eye[0], eye[1], eye[2], 1.0))
        self._draw_dice()
        self._draw_water()
        self._draw_tank()
        glfw.swap_buffers(self.window)

    def _draw_tank(self) -> None:
        if self.wireframe is None:
            return
        glDisable(GL_LIGHTING)
        glColor3f(0.78, 0.84, 0.92)
        glLineWidth(1.6)
        glBegin(GL_LINES)
        for vertex in self.wireframe:
            glVertex3f(vertex[0], vertex[1], vertex[2])
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_water(self) -> None:
        if len(self.water_faces) == 0:
            return
        glDepthMask(GL_FALSE)
        glColor4f(0.16, 0.62, 0.98, 0.78)
        glBegin(GL_TRIANGLES)
        for tri in self.water_faces:
            for idx in tri:
                normal = self.water_normals[idx]
                vertex = self.water_vertices[idx]
                glNormal3f(normal[0], normal[1], normal[2])
                glVertex3f(vertex[0], vertex[1], vertex[2])
        glEnd()
        glDepthMask(GL_TRUE)

    def _draw_dice(self) -> None:
        if len(self.dice_faces) == 0:
            return
        glColor3f(0.86, 0.79, 0.24)
        glBegin(GL_TRIANGLES)
        for tri in self.dice_faces:
            a, b, c = (self.dice_vertices[tri[0]], self.dice_vertices[tri[1]], self.dice_vertices[tri[2]])
            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-8:
                normal /= norm
            glNormal3f(float(normal[0]), float(normal[1]), float(normal[2]))
            glVertex3f(float(a[0]), float(a[1]), float(a[2]))
            glVertex3f(float(b[0]), float(b[1]), float(b[2]))
            glVertex3f(float(c[0]), float(c[1]), float(c[2]))
        glEnd()

    @staticmethod
    def _cursor_pos_callback(window, xpos: float, ypos: float) -> None:
        viewer: OpenGLViewer = glfw.get_window_user_pointer(window)
        if viewer.left_drag:
            current = np.array([xpos, ypos], dtype=np.float64)
            if viewer.last_cursor is not None:
                delta = current - viewer.last_cursor
                viewer.camera.yaw -= float(delta[0]) * 0.0055
                viewer.camera.pitch = float(np.clip(viewer.camera.pitch - delta[1] * 0.0045, -1.2, 1.2))
            viewer.last_cursor = current

    @staticmethod
    def _mouse_button_callback(window, button: int, action: int, mods: int) -> None:
        viewer: OpenGLViewer = glfw.get_window_user_pointer(window)
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                viewer.left_drag = True
                viewer.last_cursor = np.array(glfw.get_cursor_pos(window), dtype=np.float64)
            else:
                viewer.left_drag = False
                viewer.last_cursor = None

    @staticmethod
    def _scroll_callback(window, xoffset: float, yoffset: float) -> None:
        viewer: OpenGLViewer = glfw.get_window_user_pointer(window)
        viewer.camera.distance = max(6.0, viewer.camera.distance * (0.9 ** yoffset))
