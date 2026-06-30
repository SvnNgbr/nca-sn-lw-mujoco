import mujoco
import mujoco.viewer

# load xml
model = mujoco.MjModel.from_xml_path("guntherthestickman.xml")
data = mujoco.MjData(model)

# start viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()