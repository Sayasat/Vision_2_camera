# calibrate_rt.py
import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CALIB_IN  = os.path.join(SCRIPT_DIR, "calib_pairs.json")
CALIB_OUT = os.path.join(SCRIPT_DIR, "convertor_calib.json")


def estimate_rigid_transform(cam_pts: np.ndarray, rob_pts: np.ndarray):
    cam_pts = np.asarray(cam_pts, dtype=float)
    rob_pts = np.asarray(rob_pts, dtype=float)

    if cam_pts.shape != rob_pts.shape or cam_pts.ndim != 2 or cam_pts.shape[1] != 3:
        raise ValueError(f"Bad shapes: cam={cam_pts.shape}, rob={rob_pts.shape}")
    if cam_pts.shape[0] < 3:
        raise ValueError("Need at least 3 point pairs")

    c_cam = cam_pts.mean(axis=0)
    c_rob = rob_pts.mean(axis=0)

    X = cam_pts - c_cam
    Y = rob_pts - c_rob

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = c_rob - (R @ c_cam)
    return R, t


def compute_rmse(cam_pts: np.ndarray, rob_pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    pred = (cam_pts @ R.T) + t
    err = pred - rob_pts
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def main():
    print("[INFO] cwd:", os.getcwd())
    print("[INFO] script_dir:", SCRIPT_DIR)
    print("[INFO] input:", CALIB_IN)
    print("[INFO] output:", CALIB_OUT)

    if not os.path.exists(CALIB_IN):
        raise FileNotFoundError(f"Missing calib_pairs.json at: {CALIB_IN}")

    with open(CALIB_IN, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("pairs", None)
    if not isinstance(pairs, list) or len(pairs) < 3:
        raise ValueError("JSON must contain key 'pairs' with at least 3 items")

    cam_pts = []
    rob_pts = []

    for i, p in enumerate(pairs):
        if "cam" not in p or "rob" not in p:
            raise ValueError(f"Pair {i} must contain 'cam' and 'rob'")

        c = np.asarray(p["cam"], dtype=float).reshape(3)
        r = np.asarray(p["rob"], dtype=float).reshape(3)

        if not np.isfinite(c).all() or not np.isfinite(r).all():
            raise ValueError(f"Pair {i} contains NaN/Inf")

        cam_pts.append(c)
        rob_pts.append(r)

    cam_pts = np.vstack(cam_pts)
    rob_pts = np.vstack(rob_pts)

    R, t = estimate_rigid_transform(cam_pts, rob_pts)
    rmse = compute_rmse(cam_pts, rob_pts, R, t)

    out = {
        "R": R.tolist(),
        "t": t.tolist(),
        "rmse": rmse,
        "count": int(cam_pts.shape[0]),
    }

    tmp_path = CALIB_OUT + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, CALIB_OUT)

    print("[OK] Saved:", CALIB_OUT)
    print("[OK] count:", out["count"])
    print("[OK] rmse:", rmse)
    print("[OK] R:\n", np.array(out["R"]))
    print("[OK] t:\n", np.array(out["t"]))


if __name__ == "__main__":
    main()
