# Requirements:
# - Python 3.10+
# - pyglet==2.0.*, moderngl==5.*, pyrr==0.10.*
# Install: pip install pyglet moderngl pyrr

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import pyglet
from pyglet.window import key, mouse
import moderngl
import numpy as np
from pyrr import Matrix44, Vector3


# -------------------------------
# Config
# -------------------------------
WINDOW_SIZE = (1280, 720)
FOV = 70.0
NEAR = 0.1
FAR = 512.0
CHUNK_SIZE = 16
WORLD_HEIGHT = 32
GRAVITY = 22.0
JUMP_VELOCITY = 8.5
MOVE_SPEED = 7.0
SPRINT_MULT = 1.6
MOUSE_SENS = 0.12
BREAK_DISTANCE = 6.0
PLACE_DISTANCE = 6.0

# -------------------------------
# Simple Voxel Types
# -------------------------------
AIR = 0
GRASS = 1
DIRT = 2
STONE = 3
BRICK = 4

BLOCK_PALETTE = {
    GRASS: (0.38, 0.77, 0.32),
    DIRT: (0.55, 0.33, 0.21),
    STONE: (0.65, 0.65, 0.68),
    BRICK: (0.72, 0.21, 0.19),
}

# -------------------------------
# Utility
# -------------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def chunk_key(x, y, z):
    return (x // CHUNK_SIZE, y // CHUNK_SIZE, z // CHUNK_SIZE)


# -------------------------------
# World generation (very simple)
# -------------------------------
class World:
    def __init__(self):
        # dict[(cx,cy,cz)] -> numpy array (CHUNK_SIZE^3) of uint8
        self.chunks: Dict[Tuple[int, int, int], np.ndarray] = {}
        # mesh cache per chunk: vao + vbo size
        self.meshes: Dict[Tuple[int, int, int], Dict] = {}

    def generate_chunk(self, cx, cy, cz):
        if cy < 0 or cy >= math.ceil(WORLD_HEIGHT / CHUNK_SIZE):
            return np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)

        rng = random.Random((cx * 73856093) ^ (cz * 19349663))
        base_h = rng.randint(6, 12)
        heights = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.int32)
        for x in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                h = base_h
                h += int(2 * math.sin((cx * CHUNK_SIZE + x) * 0.07))
                h += int(2 * math.cos((cz * CHUNK_SIZE + z) * 0.07))
                h += rng.randint(-1, 1)
                heights[x, z] = clamp(h, 3, WORLD_HEIGHT - 1)

        data = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
        for x in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                col_h = heights[x, z]
                for y in range(CHUNK_SIZE):
                    wy = cy * CHUNK_SIZE + y
                    if wy <= col_h:
                        if wy == col_h:
                            data[x, y, z] = GRASS
                        elif wy >= col_h - 3:
                            data[x, y, z] = DIRT
                        else:
                            data[x, y, z] = STONE
        return data

    def get_chunk(self, cx, cy, cz):
        key = (cx, cy, cz)
        if key not in self.chunks:
            self.chunks[key] = self.generate_chunk(cx, cy, cz)
        return self.chunks[key]

    def get_block(self, x, y, z) -> int:
        if y < 0 or y >= WORLD_HEIGHT:
            return AIR
        cx, cy, cz = x // CHUNK_SIZE, y // CHUNK_SIZE, z // CHUNK_SIZE
        lx, ly, lz = x % CHUNK_SIZE, y % CHUNK_SIZE, z % CHUNK_SIZE
        if lx < 0:
            lx += CHUNK_SIZE
            cx -= 1
        if lz < 0:
            lz += CHUNK_SIZE
            cz -= 1
        chunk = self.get_chunk(cx, cy, cz)
        return int(chunk[lx, ly, lz])

    def set_block(self, x, y, z, block_id: int):
        if y < 0 or y >= WORLD_HEIGHT:
            return
        cx, cy, cz = x // CHUNK_SIZE, y // CHUNK_SIZE, z // CHUNK_SIZE
        lx, ly, lz = x % CHUNK_SIZE, y % CHUNK_SIZE, z % CHUNK_SIZE
        if lx < 0:
            lx += CHUNK_SIZE
            cx -= 1
        if lz < 0:
            lz += CHUNK_SIZE
            cz -= 1
        key = (cx, cy, cz)
        chunk = self.get_chunk(cx, cy, cz)
        chunk[lx, ly, lz] = block_id
        # invalidate mesh of this and neighbor chunks
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    k = (cx + dx, cy + dy, cz + dz)
                    if k in self.meshes:
                        del self.meshes[k]

    def build_chunk_mesh(self, ctx: moderngl.Context, cx, cy, cz):
        key = (cx, cy, cz)
        if key in self.meshes:
            return self.meshes[key]
        chunk = self.get_chunk(cx, cy, cz)
        if not np.any(chunk):
            self.meshes[key] = {"vao": None, "count": 0}
            return self.meshes[key]

        verts = []
        colors = []

        # cube faces
        faces = [
            ((0, 0, -1),  # back
             [(0, 0, 0), (1, 0, 0), (1, 1, 0),
              (0, 0, 0), (1, 1, 0), (0, 1, 0)]),
            ((0, 0, 1),  # front
             [(0, 0, 1), (1, 1, 1), (1, 0, 1),
              (0, 0, 1), (0, 1, 1), (1, 1, 1)]),
            ((-1, 0, 0),  # left
             [(0, 0, 0), (0, 1, 1), (0, 0, 1),
              (0, 0, 0), (0, 1, 0), (0, 1, 1)]),
            ((1, 0, 0),  # right
             [(1, 0, 0), (1, 0, 1), (1, 1, 1),
              (1, 0, 0), (1, 1, 1), (1, 1, 0)]),
            ((0, -1, 0),  # bottom
             [(0, 0, 0), (1, 0, 1), (1, 0, 0),
              (0, 0, 0), (0, 0, 1), (1, 0, 1)]),
            ((0, 1, 0),  # top
             [(0, 1, 0), (1, 1, 0), (1, 1, 1),
              (0, 1, 0), (1, 1, 1), (0, 1, 1)]),
        ]

        for lx in range(CHUNK_SIZE):
            for ly in range(CHUNK_SIZE):
                for lz in range(CHUNK_SIZE):
                    bid = int(chunk[lx, ly, lz])
                    if bid == AIR:
                        continue
                    wx, wy, wz = cx * CHUNK_SIZE + lx, cy * CHUNK_SIZE + ly, cz * CHUNK_SIZE + lz
                    base_col = BLOCK_PALETTE.get(bid, (1.0, 1.0, 1.0))
                    for (nx, ny, nz), tri in faces:
                        nb = self.get_block(wx + nx, wy + ny, wz + nz)
                        if nb == AIR:
                            # add face
                            shade = 0.85 if ny == -1 else 1.0 if ny == 1 else 0.92
                            col = tuple(min(1.0, c * shade) for c in base_col)
                            for vx, vy, vz in tri:
                                verts.extend([wx + vx, wy + vy, wz + vz])
                                colors.extend([col[0], col[1], col[2]])

        if not verts:
            self.meshes[key] = {"vao": None, "count": 0}
            return self.meshes[key]

        vbo = ctx.buffer(np.array(verts, dtype="f4").tobytes())
        cbo = ctx.buffer(np.array(colors, dtype="f4").tobytes())
        prog = get_program(ctx)
        vao = ctx.vertex_array(prog, [(vbo, "3f", "in_pos"), (cbo, "3f", "in_color")])
        self.meshes[key] = {"vao": vao, "count": len(verts) // 3}
        return self.meshes[key]

    def raycast(self, origin: Vector3, direction: Vector3, max_dist: float):
        # 3D DDA
        x, y, z = origin
        dx, dy, dz = direction
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        step_z = 1 if dz > 0 else -1

        t_max_x = ((math.floor(x) + (1 if dx > 0 else 0)) - x) / (dx if dx != 0 else 1e-6)
        t_max_y = ((math.floor(y) + (1 if dy > 0 else 0)) - y) / (dy if dy != 0 else 1e-6)
        t_max_z = ((math.floor(z) + (1 if dz > 0 else 0)) - z) / (dz if dz != 0 else 1e-6)

        t_delta_x = abs(1 / (dx if dx != 0 else 1e-6))
        t_delta_y = abs(1 / (dy if dy != 0 else 1e-6))
        t_delta_z = abs(1 / (dz if dz != 0 else 1e-6))

        bx, by, bz = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        face_normal = (0, 0, 0)

        dist = 0.0
        while dist <= max_dist:
            bid = self.get_block(bx, by, bz)
            if bid != AIR:
                return (bx, by, bz), face_normal
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    bx += step_x
                    dist = t_max_x
                    t_max_x += t_delta_x
                    face_normal = (-step_x, 0, 0)
                else:
                    bz += step_z
                    dist = t_max_z
                    t_max_z += t_delta_z
                    face_normal = (0, 0, -step_z)
            else:
                if t_max_y < t_max_z:
                    by += step_y
                    dist = t_max_y
                    t_max_y += t_delta_y
                    face_normal = (0, -step_y, 0)
                else:
                    bz += step_z
                    dist = t_max_z
                    t_max_z += t_delta_z
                    face_normal = (0, 0, -step_z)
        return None, (0, 0, 0)


# -------------------------------
# Shaders
# -------------------------------
_shader_cache = {}

def get_program(ctx: moderngl.Context):
    if "prog" in _shader_cache:
        return _shader_cache["prog"]
    prog = ctx.program(
        vertex_shader="""
        #version 330
        uniform mat4 mvp;
        in vec3 in_pos;
        in vec3 in_color;
        out vec3 v_color;
        void main() {
            v_color = in_color;
            gl_Position = mvp * vec4(in_pos, 1.0);
        }
        """,
        fragment_shader="""
        #version 330
        in vec3 v_color;
        out vec4 f_color;
        void main() {
            f_color = vec4(v_color, 1.0);
        }
        """,
    )
    _shader_cache["prog"] = prog
    return prog


# -------------------------------
# Player / Camera
# -------------------------------
@dataclass
class Player:
    pos: Vector3
    vel: Vector3
    yaw: float = 0.0
    pitch: float = 0.0
    on_ground: bool = False
    eye_height: float = 1.6
    noclip: bool = False

    def forward(self):
        cy = math.radians(self.yaw)
        cp = math.radians(self.pitch)
        return Vector3([math.cos(cp) * math.sin(cy),
                        -math.sin(cp),
                        math.cos(cp) * math.cos(cy)])

    def right(self):
        cy = math.radians(self.yaw)
        return Vector3([math.cos(cy), 0.0, -math.sin(cy)])

    def aabb(self):
        # simple capsule approximated as AABB
        w = 0.35
        h = 1.8
        minp = self.pos + Vector3([-w, 0.0, -w])
        maxp = self.pos + Vector3([w, h, w])
        return minp, maxp


def sweep_aabb(world: World, pos: Vector3, vel: Vector3, dt: float):
    # naive axis-separate collision with voxels
    remaining = vel * dt
    new_pos = Vector3(pos)

    for axis in range(3):
        delta = remaining[axis]
        if delta == 0:
            continue
        step = 1 if delta > 0 else -1
        while abs(delta) > 1e-6:
            move = clamp(delta, -0.1, 0.1)  # small steps to not tunnel
            new_pos[axis] += move
            minp, maxp = Player(new_pos, Vector3()).aabb()
            # check overlap with blocks nearby
            imin = np.floor(minp).astype(int) - 1
            imax = np.ceil(maxp).astype(int) + 1
            collided = False
            for x in range(imin[0], imax[0] + 1):
                for y in range(imin[1], imax[1] + 1):
                    for z in range(imin[2], imax[2] + 1):
                        if world.get_block(x, y, z) != AIR:
                            # block AABB
                            bmin = Vector3([x, y, z])
                            bmax = Vector3([x + 1, y + 1, z + 1])
                            overlap = not (
                                maxp[0] <= bmin[0] or minp[0] >= bmax[0] or
                                maxp[1] <= bmin[1] or minp[1] >= bmax[1] or
                                maxp[2] <= bmin[2] or minp[2] >= bmax[2]
                            )
                            if overlap:
                                # undo move and zero velocity on this axis
                                new_pos[axis] -= move
                                remaining[axis] = 0.0
                                collided = True
                                break
                    if collided:
                        break
                if collided:
                    break
            if collided:
                break
            delta -= move

    return new_pos


# -------------------------------
# Game App
# -------------------------------
class App(pyglet.window.Window):
    def __init__(self):
        super().__init__(width=WINDOW_SIZE[0], height=WINDOW_SIZE[1], caption="Voxel FPS", resizable=True)
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program = get_program(self.ctx)

        self.world = World()
        self.player = Player(
            pos=Vector3([8.0, 20.0, 8.0]),
            vel=Vector3([0.0, 0.0, 0.0]),
        )
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.set_exclusive_mouse(True)

        pyglet.clock.schedule_interval(self.update, 1 / 120.0)

    # Input
    def on_mouse_motion(self, x, y, dx, dy):
        self.player.yaw = (self.player.yaw + dx * MOUSE_SENS) % 360.0
        self.player.pitch = clamp(self.player.pitch - dy * MOUSE_SENS, -89.9, 89.9)

    def on_mouse_press(self, x, y, button, modifiers):
        fwd = self.player.forward()
        origin = self.player.pos + Vector3([0.0, self.player.eye_height, 0.0])
        if button == mouse.LEFT:
            hit, normal = self.world.raycast(origin, fwd, BREAK_DISTANCE)
            if hit:
                hx, hy, hz = hit
                self.world.set_block(hx, hy, hz, AIR)
        elif button == mouse.RIGHT:
            hit, normal = self.world.raycast(origin, fwd, PLACE_DISTANCE)
            if hit:
                nx, ny, nz = normal
                hx, hy, hz = hit
                px, py, pz = hx + nx, hy + ny, hz + nz
                # simple check to avoid placing inside player
                minp, maxp = self.player.aabb()
                if not (minp[0] <= px + 1 and maxp[0] >= px and
                        minp[1] <= py + 1 and maxp[1] >= py and
                        minp[2] <= pz + 1 and maxp[2] >= pz):
                    self.world.set_block(px, py, pz, BRICK)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()
        if symbol == key.F1:
            self.player.noclip = not self.player.noclip

    # Update
    def update(self, dt):
        spd = MOVE_SPEED * (SPRINT_MULT if self.keys[key.LSHIFT] or self.keys[key.RSHIFT] else 1.0)
        move = Vector3([0.0, 0.0, 0.0])
        fwd = self.player.forward()
        fwd[1] = 0
        if np.linalg.norm(fwd) > 1e-6:
            fwd = fwd / np.linalg.norm(fwd)
        right = self.player.right()

        if self.keys[key.W]:
            move += fwd
        if self.keys[key.S]:
            move -= fwd
        if self.keys[key.A]:
            move -= right
        if self.keys[key.D]:
            move += right

        if np.linalg.norm(move) > 1e-6:
            move = move / np.linalg.norm(move)

        if self.player.noclip:
            if self.keys[key.SPACE]:
                move[1] += 1.0
            if self.keys[key.LCTRL] or self.keys[key.RCTRL]:
                move[1] -= 1.0
            self.player.pos += move * spd * dt
        else:
            # gravity and jump
            self.player.vel[1] -= GRAVITY * dt
            if self.keys[key.SPACE] and self.player.on_ground:
                self.player.vel[1] = JUMP_VELOCITY

            # horizontal desired velocity
            desired = move * spd
            accel = 20.0
            self.player.vel[0] = (1 - math.exp(-accel * dt)) * desired[0] + math.exp(-accel * dt) * self.player.vel[0]
            self.player.vel[2] = (1 - math.exp(-accel * dt)) * desired[2] + math.exp(-accel * dt) * self.player.vel[2]

            old_pos = Vector3(self.player.pos)
            new_pos = sweep_aabb(self.world, self.player.pos, self.player.vel, dt)
            self.player.on_ground = new_pos[1] == old_pos[1] and self.player.vel[1] < 0.0
            if new_pos[1] == old_pos[1]:
                self.player.vel[1] = 0.0
            self.player.pos = new_pos

        # ensure nearby chunks are meshed
        cx, cy, cz = [int(self.player.pos[i] // CHUNK_SIZE) for i in range(3)]
        for dx in range(-2, 3):
            for dy in range(-1, 2):
                for dz in range(-2, 3):
                    self.world.build_chunk_mesh(self.ctx, cx + dx, clamp(cy + dy, 0, WORLD_HEIGHT // CHUNK_SIZE), cz + dz)

    # Draw
    def on_draw(self):
        self.clear()
        self.ctx.clear(0.53, 0.80, 0.92)
        width, height = self.get_framebuffer_size()
        aspect = width / max(1, height)

        proj = Matrix44.perspective_projection(FOV, aspect, NEAR, FAR, dtype="f4")
        eye = self.player.pos + Vector3([0.0, self.player.eye_height, 0.0])
        target = eye + self.player.forward()
        view = Matrix44.look_at(eye, target, Vector3([0.0, 1.0, 0.0]), dtype="f4")
        mvp = proj * view
        self.program["mvp"].write(mvp.astype("f4").tobytes())

        cx, cy, cz = [int(self.player.pos[i] // CHUNK_SIZE) for i in range(3)]
        rendered = 0
        for dx in range(-3, 4):
            for dy in range(-1, 2):
                for dz in range(-3, 4):
                    key = (cx + dx, clamp(cy + dy, 0, WORLD_HEIGHT // CHUNK_SIZE), cz + dz)
                    mesh = self.world.build_chunk_mesh(self.ctx, *key)
                    if mesh["vao"] and mesh["count"] > 0:
                        mesh["vao"].render(moderngl.TRIANGLES, vertices=mesh["count"])
                        rendered += 1

        # Crosshair via pyglet 2D
        x0 = width // 2
        y0 = height // 2
        crosshair_lines = [
            pyglet.shapes.Line(x0 - 8, y0, x0 + 8, y0, thickness=2, color=(255, 255, 255)),
            pyglet.shapes.Line(x0, y0 - 8, x0, y0 + 8, thickness=2, color=(255, 255, 255)),
        ]
        for line in crosshair_lines:
            line.draw()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.ctx.viewport = (0, 0, width, height)


if __name__ == "__main__":
    App()
    pyglet.app.run()
