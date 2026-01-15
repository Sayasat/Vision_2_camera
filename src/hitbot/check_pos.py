from hitbot.hitbot_move_python import HitbotMove
import time

robot = HitbotMove(74)  # твой ID или IP
robot.safe_move_xyz(50,0,0,0,50)
robot.safe_move_xyz(24.67799999999997, -252.548, -0,0,50)
robot.safe_move_xyz(24.67799999999997, -252.548, -301.0,0,50)
# robot.safe_move_xyz(30.28, -305.44, -167.00,0,50)

    # robot.new_movej_xyz_lr(167, 200, -100,0,70,0,1)
    # robot.wait_stop()
    # robot.new_movej_xyz_lr(167, -331, -250,0,70,0,1)
    # robot.wait_stop()

    # robot.new_movej_xyz_lr(325, -431, -315,0,70,0,1)
    # robot.wait_stop()
