import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os

# Lade das Modell
model = mujoco.MjModel.from_xml_path("guntherthestickman.xml")
data = mujoco.MjData(model)

# Erstelle einen Ordner für die Bilder
os.makedirs("frames", exist_ok=True)

# Rendere 5 Sekunden bei 60 FPS (300 Frames)
duration = 5.0  # Sekunden
fps = 60
num_frames = int(duration * fps)

# Kamera-Einstellungen: falls im Modell Kameras definiert sind, verwende
# die erste Kamera (Index 0). Andernfalls benutze den freien Standard-
# Kameramodus (camera_id=-1) und MuJoCo's default free camera.
camera_id = 0 if model.ncam > 0 else -1

# Rendere Frames mit Offscreen-Renderer
# Wähle eine Renderauflösung (kann angepasst werden)
render_width = 640
render_height = 480

frame_files = []
with mujoco.Renderer(model, render_height, render_width) as renderer:
    for i in range(num_frames):
        # Simuliere einen Schritt und berechne Vorwärtskinematik
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        # Aktualisiere die Szene und rendere in ein Array
        renderer.update_scene(data, camera_id)
        out = np.zeros((render_height, render_width, 3), np.uint8)
        img = renderer.render(out=out)

        # MuJoCo liefert RGB, OpenCV erwartet BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fname = f"frames/frame_{i:04d}.png"
        cv2.imwrite(fname, img_bgr)
        frame_files.append(fname)

# Erstelle ein Video aus den Frames (bekannte Auflösung verwenden)
video = cv2.VideoWriter(
    "stickman_standing.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (render_width, render_height),
)

for frame_file in frame_files:
    frame = cv2.imread(frame_file)
    video.write(frame)

video.release()

# Lösche die temporären Frames (optional)
for frame_file in frame_files:
    os.remove(frame_file)
os.rmdir("frames")

print("Video wurde als 'stickman_standing.mp4' gespeichert!")