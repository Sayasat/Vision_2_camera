import csv
import convertor 

rx, ry, rz = convertor.camera_to_robot_xyz(451.980, -31.660, 694.000)
print(f"Robot coordinates: X={rx}, Y={ry}, Z={rz}")