import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os

model = mujoco.MjModel.from_xml_path("guntherthestickman.xml")
data = mujoco.MjData(model)

os.makedirs("frames", exist_ok=True)

duration = 5.0  # seconds
fps = 60
num_frames = int(duration * fps)

camera_id = 0 if model.ncam > 0 else -1

render_width = 640
render_height = 480

# def of angels for upright stand
target_qpos = np.array([
    0.0,   # right_shoulder_z
    -10.0, # right_shoulder_y (leicht nach hinten geneigt)
    0.0,   # right_elbow
    0.0,   # right_knee
    -10.0, # right_hip (leicht nach hinten geneigt)
    0.0,   # left_shoulder_z
    -10.0, # left_shoulder_y
    0.0,   # left_elbow
    0.0,   # left_knee
    -10.0, # left_hip
])

# PD-Regler-Parameter
kp = 1.0  # Prop
kd = 0.10   # Deriv


frame_files = []
with mujoco.Renderer(model, render_height, render_width) as renderer:
    for i in range(num_frames):
        for j in range(model.nu):
            qpos = data.qpos[model.actuator_trnid[j, 0]] 
            qvel = data.qvel[model.actuator_trnid[j, 0]]
            data.ctrl[j] = kp * (target_qpos[j] - qpos) - kd * qvel

        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera_id)
        out = np.zeros((render_height, render_width, 3), np.uint8)
        img = renderer.render(out=out)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fname = f"frames/frame_{i:04d}.png"
        cv2.imwrite(fname, img_bgr)
        frame_files.append(fname)

video = cv2.VideoWriter(
    "stickman_stable_stand.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (render_width, render_height),
)

for frame_file in frame_files:
    frame = cv2.imread(frame_file)
    video.write(frame)

video.release()

# cleanup
for frame_file in frame_files:
    os.remove(frame_file)
os.rmdir("frames")

print("Video wurde als 'stickman_stable_stand.mp4' gespeichert!")


'''
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        # PD-Regler: calc control for each actuator
        for i in range(model.nu):
            # ist angle and speed
            qpos = data.qpos[model.actuator_trnid[i, 0]]  # angle
            qvel = data.qvel[model.actuator_trnid[i, 0]]  # speed

            # control = kp * (target - current) - kd * speed
            data.ctrl[i] = kp * (target_qpos[i] - qpos) - kd * qvel

        viewer.sync()
'''