# convertor.py
import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(SCRIPT_DIR, "convertor_calib.json")

if os.path.exists(CALIB_PATH):
    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    R = np.array(d["R"], dtype=float)
    t = np.array(d["t"], dtype=float)
else:
    # fallback (твой старый костыль)
    cam_ref = np.array([312, 100, 620.0], dtype=float)
    rob_ref = np.array([100, -350, -236.0], dtype=float)
    R = np.eye(3, dtype=float)
    R[1, 1] = -1
    t = rob_ref - (R @ cam_ref)

def camera_to_robot_xyz(cam_x, cam_y, cam_z):
    cam_xyz = np.array([cam_x, cam_y, cam_z], dtype=float)
    rob_xyz = (R @ cam_xyz) + t
    return rob_xyz
