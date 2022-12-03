import glfw
from OpenGL.GL import *
from editor import Editor, Camera, Part
import numpy as np

def main():
  window_size = (1400, 800)
  def on_resize(window, size_x, size_y):
    nonlocal window_size
    window_size = (size_x, size_y)
    glViewport(0, 0, size_x, size_y)

  if not glfw.init():
    return

  window = glfw.create_window(*window_size, "Lego editor", None, None)
  glfw.set_framebuffer_size_callback(window, on_resize)
  if not window:
    glfw.terminate()
    return
  glfw.make_context_current(window)
  glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
  glClearColor(255, 255, 255, 0)

  camera = Camera()
  editor = Editor(camera)

  prevCursorPos = (-1, -1)
  yawPitch = (0, 0)
  def cursorPosCallback(window, xpos, ypos):
    nonlocal prevCursorPos
    nonlocal yawPitch
    nonlocal camera
    if prevCursorPos[0] < 0:
      prevCursorPos = (xpos, ypos)
      return
    sensitivity = 0.1
    delta = ((xpos - prevCursorPos[0]) * -sensitivity, (ypos - prevCursorPos[1]) * -sensitivity)
    yawPitch = (yawPitch[0] + delta[0], yawPitch[1] + delta[1])
    if yawPitch[1] > 89:
      yawPitch = (yawPitch[0], 89)
    if yawPitch[1] < -89:
      yawPitch = (yawPitch[0], -89)
    camera.set_euler_angles(yawPitch)
    prevCursorPos = (xpos, ypos)
  glfw.set_cursor_pos_callback(window, cursorPosCallback)
  def keyCallback(window, key, scancode, action, mods):
    nonlocal camera
    nonlocal editor
    movement_speed = 0.1
    if action != glfw.PRESS and action != glfw.REPEAT:
      return
    if key == glfw.KEY_W:
      camera.pos -= camera.forward * movement_speed
    if key == glfw.KEY_S:
      camera.pos += camera.forward * movement_speed
    camera_right = np.cross(camera.forward, np.array([0, 1, 0], dtype="float32"))
    if key == glfw.KEY_D:
      camera.pos += camera_right * movement_speed
    if key == glfw.KEY_A:
      camera.pos -= camera_right * movement_speed
    if key == glfw.KEY_LEFT_SHIFT:
      camera.pos[1] -= movement_speed
    if key == glfw.KEY_LEFT_CONTROL:
      camera.pos[1] += movement_speed
    if key == glfw.KEY_UP:
      editor.light_positions[0] += camera.forward * movement_speed
    if key == glfw.KEY_DOWN:
      editor.light_positions[0] -= camera.forward * movement_speed
    if key == glfw.KEY_RIGHT:
      editor.light_positions[0] -= camera_right * movement_speed
    if key == glfw.KEY_LEFT:
      editor.light_positions[0] += camera_right * movement_speed
    if key == glfw.KEY_RIGHT_SHIFT:
      editor.light_positions[0][1] += movement_speed
    if key == glfw.KEY_RIGHT_CONTROL:
      editor.light_positions[0][1] -= movement_speed
    if key == glfw.KEY_LEFT_BRACKET:
      editor.light_dim -= 0.02
    if key == glfw.KEY_RIGHT_BRACKET:
      editor.light_dim += 0.02
    if action != glfw.PRESS:
      return
    if key == glfw.KEY_R:
      editor.active_part.rotation += 1
    if key == glfw.KEY_I:
      editor.active_part.pos[0] += 1
    if key == glfw.KEY_K:
      editor.active_part.pos[0] -= 1
    if key == glfw.KEY_L:
      editor.active_part.pos[2] += 1
    if key == glfw.KEY_J:
      editor.active_part.pos[2] -= 1
    if key == glfw.KEY_U:
      editor.active_part.pos[1] += 1
    if key == glfw.KEY_O:
      editor.active_part.pos[1] -= 1
    if key == glfw.KEY_P:
      editor.fix()
    if key == glfw.KEY_C:
      editor.active_part.color += 1
    if key == glfw.KEY_F:
      editor.active_type += 1
      editor.active_type %= Part.part_types
    if key == glfw.KEY_Z:
      editor.undo()
  glfw.set_key_callback(window, keyCallback)


  glEnable(GL_DEPTH_TEST)
  
  while not glfw.window_should_close(window):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    editor.render(window_size);

    glfw.swap_buffers(window)
    glfw.poll_events()
  glfw.terminate()

if __name__ == "__main__":
  main()
