""" Demonstrates the Franka Emika Robot System model for MuJoCo """
import time
from threading import Thread
import mujoco
import glfw
import numpy as np


class Demo:

    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("world.xml")
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.run = True
        self.gripper(True)
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i-1]
        mujoco.mj_forward(self.model, self.data)
        Thread(target=self.render).start()
        Thread(target=self.step).start()

    def gripper(self, open = True):
        self.data.actuator("pos_panda_finger_joint1").ctrl = (0.04, 0)[not open]
        self.data.actuator("pos_panda_finger_joint2").ctrl = (0.04, 0)[not open]

    def control(self, xpos_d, xquat_d):
        xpos = self.data.body("panda_hand").xpos
        xquat = self.data.body("panda_hand").xquat
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bodyid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

        error = np.zeros(6)
        error[:3] = xpos_d-xpos
        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res

        J = np.concatenate((jacp, jacr))
        v = J@self.data.qvel
        for i in range(1, 8):
            dofadr = self.model.joint(f"panda_joint{i}").dofadr
            self.data.actuator(f"panda_joint{i}").ctrl = self.data.joint(f"panda_joint{i}").qfrc_bias 
            self.data.actuator(f"panda_joint{i}").ctrl += J[:,dofadr].T@np.diag(self.K)@error
            self.data.actuator(f"panda_joint{i}").ctrl -= J[:,dofadr].T@np.diag(2*np.sqrt(self.K))@v

    def step(self):
        import copy
        xpos0 = copy.copy(self.data.body("panda_hand").xpos)
        xpos_d = xpos0
        xquat0 = copy.copy(self.data.body("panda_hand").xquat)
        down = list(np.linspace(-.45, 0, 2000))
        up = list(np.linspace(0, -.45, 2000))
        state = "down"
        while self.run:
            if state == "down":
                if len(down):
                    xpos_d = xpos0 + [0, 0, down.pop()]
                else:
                    state = "grasp"
            elif state == "grasp":
                self.gripper(False)
                state = "up"
            elif state == "up":
                if len(up):
                    xpos_d = xpos0 + [0, 0, up.pop()]
            self.control(xpos_d, xquat0)
            mujoco.mj_step(self.model, self.data)
            time.sleep(1e-3)

    def render(self):
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(640, 480, "Demo", None, None)
        glfw.make_context_current(window)
        context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_100.value)
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, 640, 480)
        while not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width = w
            viewport.height = h
            mujoco.mjv_updateScene(
                self.model, self.data, opt, pert,
                self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
            mujoco.mjr_render(viewport, self.scene, context)
            time.sleep(1/30)
            glfw.swap_buffers(window)
            glfw.poll_events()
        self.run = False

if __name__ == "__main__":
    Demo()
