import numpy
from OpenGL.GL import *
from numpy import cos, sin

from shader import Shader
import os
import numpy as np
from ctypes import c_void_p
import json
from mesh import Mesh
import math
import sys

class Editor:
    def __init__(self, camera):
        self.target_file = sys.argv[1] if len(sys.argv) > 1 else "tmp.json"
        
        Part.static_init()
        self.program = Shader("shader.vert", "shader.frag")
        glUseProgram(self.program.program)
        self.model_location = glGetUniformLocation(self.program.program, "model")
        self.model_normal_location = glGetUniformLocation(self.program.program, "model_norm");
        self.camera_location = glGetUniformLocation(self.program.program, "camera")
        self.part_color_location = glGetUniformLocation(self.program.program, "partColor")
        self.light_position_location = glGetUniformLocation(self.program.program, "lightPosition")
        self.light1_position_location = glGetUniformLocation(self.program.program, "light1Position")
        self.light_status = glGetUniformLocation(self.program.program, "lightStatus")
        self.light1_status = glGetUniformLocation(self.program.program, "light1Status")
        self.dim_location = glGetUniformLocation(self.program.program, "dim")
        self.camera = camera
        self.rotate_lights = 0
        self.rotate_lights_angle = 0
        self.rotate_lights_radius = 5
        self.active_part = Part()
        self.active_type = 0
   
        self.parts = [[] for i in range(Part.part_types)]
        self.part_history = []

        if os.path.exists(self.target_file):
            with open(self.target_file) as file:
                editor_save = json.load(file)
            self.parts = editor_save["parts"]
            self.parts = [[Part.from_json(part) for part in parts] for parts in self.parts]
            self.part_history = editor_save["part_history"]

        self.colors = np.array([[222, 0, 13],
                       [0, 87, 168],
                       [242, 205, 55],
                       [24, 70, 50],
                       [255, 255, 255],
                       [5, 19, 29]], dtype="float32") / 256

        # self.light_pos = np.array([0, 10, 0], dtype="float32")
        ((l1x, l1y, l1z), (l2x, l2y, l2z)) = ((10, 0, 10), (-10, 0, -10))
        self.light_positions = np.array([[l1x, l1y, l1z], [l2x,l2y,l2z]], dtype="float32")
        self.light_statuses = [1.0, 1.0]
        self.light_dim = 0.98

        self.cube_vao = glGenVertexArrays(1)
        cubeBuffer = glGenBuffers(1)
        glBindVertexArray(self.cube_vao)
        glBindBuffer(GL_ARRAY_BUFFER, cubeBuffer)
        glBufferData(GL_ARRAY_BUFFER, np.array([
                1, 0, 0,
                -1, 1, -1,     # Front-top-left
                1, 1, -1,      # Front-top-right
                -1, -1, -1,    # ront-bottom-left
                1, -1, -1,     # Front-bottom-right
                1, -1, 1,      # Back-bottom-right
                1, 1, -1,      # Front-top-right
                1, 1, 1,       # Back-top-right
                -1, 1, -1,     # Front-top-left
                -1, 1, 1,      # Back-top-left
                -1, -1, -1,    # Front-bottom-left
                -1, -1, 1,     # Back-bottom-left
                1, -1, 1,      # Back-bottom-right
                -1, 1, 1,      # Back-top-left
                1, 1, 1        # Back-top-right
            ], dtype="float32") / 4, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, c_void_p(12))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(1)
        
        
    def save(self):
        with open(self.target_file,mode="w") as file:
            json.dump({"parts": [[part.to_dict() for part in parts] for parts in self.parts], "part_history": self.part_history}, file)
    def render(self, window_size):
        glUseProgram(self.program.program)
        if self.rotate_lights:
            print('render', self.rotate_lights_angle)
            ((l1x, l1y, l1z), (l2x, l2y, l2z)) = (
                (
                    self.rotate_lights_radius * cos(self.rotate_lights_angle)+self.rotate_lights_radius*sin(self.rotate_lights_angle),
                    0,
                    -self.rotate_lights_radius*sin(self.rotate_lights_angle)+self.rotate_lights_radius * cos(self.rotate_lights_angle)
                ),
                (
                    - (self.rotate_lights_radius * cos(self.rotate_lights_angle) + self.rotate_lights_radius * sin(self.rotate_lights_angle)),
                    0,
                    - (-self.rotate_lights_radius * sin(self.rotate_lights_angle) + self.rotate_lights_radius * cos(self.rotate_lights_angle))
                )
            )
            self.rotate_lights_angle+=numpy.pi/1800
            self.light_positions = np.array([[l1x, l1y, l1z], [l2x, l2y, l2z]], dtype="float32")
        glUniformMatrix4fv(self.camera_location, 1, GL_FALSE, self.camera.get_matrix(window_size[0] / window_size[1]))
        glUniform3fv(self.light_position_location, 1, self.light_positions[0])
        glUniform3fv(self.light1_position_location, 1, self.light_positions[1])
        glUniform1f(self.dim_location, self.light_dim)
        glUniform1f(self.light_status, self.light_statuses[0])
        glUniform1f(self.light1_status, self.light_statuses[1])

        glBindVertexArray(self.cube_vao)
        glUniform3f(self.part_color_location, 100, 100, 100)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 14)
        for part_type, parts in enumerate(self.parts):
            Part.render_many(part_type, parts, self)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        Part.render_many(self.active_type, [self.active_part], self)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        return

    def fix(self):
        active_rect = self.active_part.rect(self.active_type)
        def intersect(rect1, rect2):
            return not (rect1[1] <= rect2[0] or rect1[0] >= rect2[1] or rect1[3] <= rect2[2] or rect1[2] >= rect2[3] or rect1[5] <= rect2[4] or rect1[4] >= rect2[5])

        for part_type, parts in enumerate(self.parts):
            for part in parts:
                if intersect(active_rect, part.rect(part_type)):
                    print("not intersect")
                    return 0
        
        self.parts[self.active_type].append(self.active_part)
        self.part_history.append(self.active_type)
        self.active_part = Part(self.active_part.pos.copy())
        self.save()
        return 1
    def undo(self):
        if len(self.part_history) == 0:
            return
        self.parts[self.part_history[-1]].pop()
        self.part_history.pop()
        self.save()

class Camera:
    def lookAt(position, target, up):
        
        forward = np.subtract(target, position)
        forward = np.divide( forward, np.linalg.norm(forward) )

        right = np.cross( forward, up )
        
        # if forward and up vectors are parallel, right vector is zero; 
        #   fix by perturbing up vector a bit
        if np.linalg.norm(right) < 0.001:
            epsilon = np.array( [0.001, 0, 0] )
            right = np.cross( forward, up + epsilon )
            
        right = np.divide( right, np.linalg.norm(right) )
        
        up = np.cross( right, forward )
        up = np.divide( up, np.linalg.norm(up) )
        
        return np.array([[right[0], up[0], -forward[0], position[0]], 
                         [right[1], up[1], -forward[1], position[1]], 
                         [right[2], up[2], -forward[2], position[2]],
                         [0, 0, 0, 1]]) .T
    def perspective(field_of_view_y, aspect, z_near, z_far):

        fov_radians = math.radians(field_of_view_y)
        f = math.tan(fov_radians/2)

        a_11 = 1/(f*aspect)
        a_22 = 1/f
        a_33 = -(z_near + z_far)/(z_near - z_far)
        a_34 = -2*z_near*z_far/(z_near - z_far)

        # a_33 = -(z_far + z_near)/(z_far - z_near)
        # a_34 = 2*z_far*z_near/(z_far - z_near)

        perspective_matrix = np.matrix([
            [a_11, 0, 0, 0],       
            [0, a_22, 0, 0],       
            [0, 0, a_33, a_34],    
            [0, 0, -1, 0]          
        ]).T

        return perspective_matrix
    def __init__(self):
        self.pos = np.array([0, 0, -1.8], dtype="float32")
        self.forward = np.array([0, 0, 1], dtype="float32")
    def get_matrix(self, aspect):
        right = np.cross(self.forward, np.array([0, 1, 0], dtype="float32"))
        up = np.cross(self.forward, right)
        mat = np.linalg.inv(Camera.lookAt(self.pos, self.pos + self.forward, np.array([0, 1, 0])))
        #mat = Camera.perspective(60, aspect, 0.1, 100) @ mat
        return mat
    def set_euler_angles(self, yawPitch):
        ypr = np.radians(yawPitch)
        x = np.cos(ypr[0]) * np.cos(ypr[1])
        y = np.sin(ypr[1])
        z = np.sin(ypr[0]) * np.cos(ypr[1])
        self.forward = np.array([x, y, z], dtype="float32")
        self.forward /= np.linalg.norm(self.forward)



def centroid(vertexes):
    _x_list = [vertex[0] for vertex in vertexes]
    _y_list = [vertex[1] for vertex in vertexes]
    _z_list = [vertex[2] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    _z = sum(_z_list) / _len
    return(_x, _y, _z)


def in_shadow(part, part_types, light_position, parts):
    sizes = part.sizes[part_types]
    rect_sample = [
        [0, 0, 0],
        [0, -sizes[1], 0],
        [0, -sizes[1], sizes[2]],
        [0, 0, sizes[2]],
        [-sizes[0], 0, 0],
        [-sizes[0], -sizes[1], 0],
        [-sizes[0], -sizes[1], sizes[2]],
        [-sizes[0], 0, sizes[2]]
    ]
    rect = [[_[0] + part.pos[0], _[1] + part.pos[1], _[2] + part.pos[2]] for _ in rect_sample]
    _surface_points = [(pow(point[0] - light_position[0], 2) + pow(point[1] - light_position[1], 2) + pow(
        point[2] - light_position[2], 2), i) for i, point in enumerate(rect)]
    _surface_points.sort()
    surface = [rect[_point[1]] for _point in _surface_points[:3]]
    center_point = centroid(surface)  # координаты нормали к поверхности
    parametric_equation = center_point - light_position[0]  # i,j,k equation
    for _ in parts:
        for _part in _:
            if part.pos == _part.pos:
                break
            rect = [[_[0] + _part.pos[0], _[1] + _part.pos[1], _[2] + _part.pos[2]] for _ in rect_sample]
            for surface in rect:
                if abs(-(parametric_equation[0]*surface[0] + parametric_equation[1]+surface[1])/parametric_equation[2]-surface[2]) < 0.01 or \
                    abs(-(parametric_equation[0] * surface[0] + parametric_equation[2] + surface[2]) /parametric_equation[1] - surface[1]) < 0.01 or \
                    abs(-(parametric_equation[1] * surface[1] + parametric_equation[2] + surface[2]) / parametric_equation[0] - surface[0]) < 0.01:
                    return True
    return False
class Part:
    initialized = False
    part_types = 0
    vertex_arrays = []
    buffers = []
    elements_counts = []
    sizes = []
    faces = []
    def static_init():
        if Part.initialized:
            return
        Part.initialized = True

        with open(os.path.join(os.path.dirname(__file__), "parts/list.json")) as file:
          parts = json.load(file)
        Part.part_types = len(parts)
        alloc = len(parts) + 1
        Part.vertex_arrays = glGenVertexArrays(alloc)
        Part.buffers = glGenBuffers(alloc)

        Part.elements_counts = [0 for i in range(alloc)]
        Part.faces = [0 for i in range(alloc)]
        Part.sizes = [0 for i in range(alloc)]

        for part_type, part in enumerate(parts):
            mesh = Mesh(os.path.join(os.path.dirname(__file__), "parts", part["file"]), Part.vertex_arrays[part_type], Part.buffers[part_type], part["size"], part["pin"])
            Part.sizes[part_type] = part["size"]
            Part.elements_counts[part_type] = mesh.count
            Part.faces[part_type] = mesh.result

    def render_many(part_type, parts, editor):
        glBindVertexArray(Part.vertex_arrays[part_type])
        for light_position in editor.light_positions:

            for part in parts:
                # algorithm. If faces vector intersect more than one part, any other part not highlighted
                model = part.model_matrix()
                glUniformMatrix4fv(editor.model_location, 1, GL_FALSE, model)
                # if not in_shadow(part, part_type, light_position,editor.parts):
                #     #set light_position to 0
                #     glUniform1f(editor.light_status, 0.1)
                # else:
                #     #set dim to 0.2
                #     glUniform1f(editor.light_status, 1)
                glUniformMatrix3fv(editor.model_normal_location, 1, GL_FALSE, np.linalg.inv(model[:-1,:-1]).transpose())
                glUniform3fv(editor.part_color_location, 1, editor.colors[part.color % len(editor.colors)])

                glDrawArrays(GL_TRIANGLES, 0, Part.elements_counts[part_type])

    def __init__(self, pos=[0, 0, 0]):
        self.pos = pos
        self.rotation = 0
        self.color = 0

    def from_json(json):
        obj = Part(json["pos"])
        obj.rotation = json["rotation"]
        obj.color = json["color"]
        return obj

    def to_dict(self):
        return {"pos": self.pos, "rotation": self.rotation, "color": self.color}

    def model_matrix(self):
        model = np.identity(4, dtype="float32")
        model[0, 0] = math.cos(math.pi / 2 * self.rotation)
        model[2, 0] = math.sin(math.pi / 2 * self.rotation)
        model[0, 2] = -model[2, 0]
        model[2, 2] = model[0, 0]
        model[-1, :-1] = self.pos
        model[-1, 1] *= 0.4
        return model

    def rect(self, part_type):
        size = np.array(Part.sizes[part_type], dtype="float32")
        size = self.model_matrix()[:-1,:-1] @ size
        x2 = self.pos[0] + size[0]
        y2 = self.pos[1] + size[1]
        z2 = self.pos[2] + size[2]
        return (min(self.pos[0], x2), max(self.pos[0], x2), min(self.pos[1], y2), max(self.pos[1], y2), min(self.pos[2], z2), max(self.pos[2], z2))
