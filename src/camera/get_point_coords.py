import cv2
import numpy as np
from driver import Gemini335LECamera
import csv

current_pc = None
clicked_points = []  

import csv

def mouse_callback(event, x, y, flags, param):
    global current_pc, clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and current_pc is not None:
        if 0 <= y < current_pc.shape[0] and 0 <= x < current_pc.shape[1]:
            X, Y, Z = current_pc[y, x]
            if Z > 0 and not np.isnan(Z):
                print(f"Clicked pixel ({x}, {y}) → X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m")
                clicked_points.append((x, y))
                # Save to CSV
                with open("clicked_points.csv", "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([x, y, X, Y, Z])
            else:
                print(f"Clicked pixel ({x}, {y}) → Invalid depth")


if __name__ == "__main__":
    cam = Gemini335LECamera(serial_number="CPE345P0008A")
    cam.start()
    print("Click on the image to get 3D coordinates. Press 'q' to quit.")
    cv2.namedWindow("Color")
    cv2.setMouseCallback("Color", mouse_callback)

    try:
        while True:
            color_img, pc = cam.get_pointcloud()
            if color_img is not None:

                for pt in clicked_points:
                    cv2.drawMarker(color_img, pt, color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                cv2.imshow("Color", color_img)

            if pc is not None:
                current_pc = pc

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")
