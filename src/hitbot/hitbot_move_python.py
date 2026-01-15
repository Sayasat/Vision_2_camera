from math import isclose
import time
from hitbot.hitbot_interface import HitbotInterface
import csv

# Path to CSV file containing object coordinates
csv_path = "/home/sayasn/Robot/New_work/hitbot_ws_Legion/hitbot_ws/src/camera_node/camera_node/cv2_learning/hitbot/coordinates.csv"

# Robot motion constants
SAFE_Z     = -80      # Safe height above objects (to avoid collisions)
SAFE_R     = 0        # Default rotation
FLIP_SPD   = 100      # Speed for flipping hand orientation
Z_MIN_SAFE = -220     # Lowest safe Z value
ROUGHLY    = 1.0      # Positioning tolerance

class HitbotMove:
    """
    Class to control the Hitbot manipulator for a pick-and-place sequence.
    Reads object positions from CSV file and moves them to predefined locations.
    """

    # Predefined byte sequences for suction control
    SUCTION_ON      = bytes.fromhex('011000020002040001000023b6')
    SUCTION_RELEASE = bytes.fromhex('0110000200020400020000d3b6')

    def __init__(self, ID):
        # Connect to the Hitbot robot
        self.robot = HitbotInterface(ID)
        self.robot.net_port_initial()
        self._startup_pins()
        time.sleep(2)  # Wait for connection to stabilize

        if self.robot.is_connect() != 1:
            print("ERROR: Robot not connected.")
            exit()

        # Ensure all joints are safe before starting
        for j in range(1, 5):
            self.robot.check_joint(j, False)
        self.robot.unlock_position()
        self.robot.initial(5, 410.0)

        self.robot.hand = -1  # Current hand orientation (left/right)
        self.last_target = None  # Last picked object location

        self.iterations_remaining = 0

        # Predefined placing positions
        self.place_positions = [
            (160.0, 230.0, -160.0),
            (160.0, 370.0, -160.0),
            (230.0, 370.0, -160.0),
            (230.0, 230.0, -160.0)
        ]
        self.current_place_index = 0
        
        # Object coordinates from CSV
        self.object_coordinates = self.read_coordinates_from_csv()
        self.current_object_index = 0
 
        print("Hitbot connected and initialized. Ready for pick-and-place operations.")

    def read_coordinates_from_csv(self):
        """Read object coordinates from CSV file."""
        coordinates = []
        try:
            with open(csv_path, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if len(row) >= 3:
                        try:
                            x = float(row[0])
                            y = float(row[1])
                            z = float(row[2])
                            coordinates.append((x, y, z))
                        except ValueError:
                            print(f"WARNING: Skipping invalid row: {row}")
            print(f"Read {len(coordinates)} coordinates from CSV")
        except FileNotFoundError:
            print(f"ERROR: CSV file not found at {csv_path}")
        except Exception as e:
            print(f"ERROR: Failed to read CSV file: {e}")
        
        return coordinates

    def set_iterations(self, iterations):
        """Set number of iterations (objects to move)."""
        self.iterations_remaining = iterations
        print(f"New number of iterations: {self.iterations_remaining}")

    def send_command(self, msg, label):
        """Send a raw command (e.g., suction on/off) to the robot."""
        print(f"Sending {label}: {msg.hex()}")
        time.sleep(0.5)
        if self.robot.com485_send(msg, len(msg)) == 1:
            print(f"{label} command sent.")
        else:
            print(f"WARNING: {label} command failed.")

    def wait_until_target_reached(self):
        """Block until the robot reaches the target position."""
        self.robot.set_allow_distance_at_target_position(0.5, 0.5, 0.5, 9999)
        print("Waiting until robot reaches the target...")
        while self.robot.is_robot_goto_target() != 1:
            time.sleep(0.1)
        print("Target reached.")

    def safe_move_xyz(self, x, y, z, r, speed=None):
        """
        Move the robot to a given XYZ position safely, trying both hand orientations if needed.
        """

        def _move(lr):
            result = self.robot.new_movej_xyz_lr(x, y, z, r, speed, ROUGHLY, lr)
            print(f"Attempt with lr={lr}, speed={speed}, roughly={ROUGHLY}: result = {result}")
            return result

        lr_current = getattr(self.robot, "hand", -1)

        # Try moving with current hand orientation
        if _move(lr_current) == 1:
            self.wait_until_target_reached()
            return True

        # If failed, try flipping the hand orientation
        print(f"WARNING: Trying to flip from lr={lr_current} to {-lr_current}")
        self.robot.new_movej_xyz_lr(self.robot.x, self.robot.y, SAFE_Z, r, FLIP_SPD, ROUGHLY, lr_current)
        self.wait_until_target_reached()

        lr_other = -lr_current
        self.robot.change_attitude(FLIP_SPD)

        if _move(lr_other) == 1:
            self.robot.hand = lr_other
            self.wait_until_target_reached()
            return True

        raise RuntimeError("Target unreachable in either hand system.")

    def pick_and_place(self, pick_xyz, place_xyz, z_safe=-15.0, carrying_speed=75, moving_speed=100, r=0.0):
        """
        Full pick-and-place sequence:
        1. Move to safe position above pick point
        2. Move down to pick point
        3. Turn suction ON
        4. Move up to safe height
        5. Move to above place position
        6. Move down to place point
        7. Turn suction OFF
        8. Return to home position
        """
        x_pick, y_pick, z_pick = pick_xyz
        x_place, y_place, z_place = place_xyz
        total_start = time.time()

        print("--- Pick and Place Sequence Start ---")
        self.safe_move_xyz(x_pick, y_pick, z_safe, r, moving_speed)
        self.safe_move_xyz(x_pick, y_pick, z_pick, r, moving_speed)
        self.wait_until_target_reached()

        self.send_command(self.SUCTION_ON, "Suction ON")
        time.sleep(0.2)

        self.safe_move_xyz(x_pick, y_pick, z_safe, r, carrying_speed)
        self.wait_until_target_reached()

        self.safe_move_xyz(x_place, y_place, z_safe, r, carrying_speed)

        self.safe_move_xyz(x_place, y_place, z_place, r, carrying_speed)
        self.wait_until_target_reached()

        self.send_command(self.SUCTION_RELEASE, "Suction RELEASE")
        time.sleep(0.2)

        self.safe_move_xyz(x_place, y_place, z_safe, r, moving_speed)
        self.wait_until_target_reached()

        # Go to home position
        self.safe_move_xyz(50.0, 0.0, z_safe, r, moving_speed)
        self.wait_until_target_reached()

        total_time = time.time() - total_start
        print(f"✅ Pick-and-place total time: {total_time:.2f}s")

    def process_next_object(self):
        """Process the next object from the CSV file."""
        if self.iterations_remaining <= 0:
            print("No more iterations. Use set_iterations() to set new iterations.")
            return False

        if self.current_object_index >= len(self.object_coordinates):
            print("No more objects in CSV file.")
            return False

        x, y, z = self.object_coordinates[self.current_object_index]
        r = 0.0

        # Ignore if same target as last time
        if self.last_target and all((
            isclose(x, self.last_target[0], abs_tol=0.1),
            isclose(y, self.last_target[1], abs_tol=0.1),
            isclose(z, self.last_target[2], abs_tol=0.1))):
            print("Target is the same as last one. Ignoring.")
            self.current_object_index += 1
            return True

        self.last_target = (x, y, z)
        print(f"Processing target point: x={x}, y={y}, z={z}")

        try:
            # Select place position (loop through predefined ones)
            if self.current_place_index < len(self.place_positions):
                place_xyz = self.place_positions[self.current_place_index]
            else:
                print("⚠️ No more predefined place positions. Reusing the last one.")
                place_xyz = self.place_positions[-1]

            self.pick_and_place((x, y, z), place_xyz)

            self.iterations_remaining -= 1
            self.current_object_index += 1
            self.current_place_index += 1

            print(f"Iterations left: {self.iterations_remaining}")
            return True
            
        except RuntimeError as e:
            print(f"ERROR: Pick-and-place failed: {e}")
            return False

    def run_all_objects(self, iterations=None):
        """Process all available objects."""
        if iterations is not None:
            self.set_iterations(iterations)
        
        while self.process_next_object():
            time.sleep(1)  # Small delay between operations
    


    def get_in(self, pin):
        return self.robot.get_digital_in(pin)

    def set_output(self, pin: int, on: bool):
        while True:
            self.robot.set_digital_out(pin, 1 if on else 0)
            st = self.robot.get_digital_out(pin)
            if st == (1 if on else 0):
                print(f"DO{pin} = {st}")
                break

    def check_all_outputs(self):
        output = []
        for i in range(0, 8):     
            v = self.robot.get_digital_out(i)
            output.append(v)
        print(output)

    def check_all_inputs(self):
        inputs = []
        for i in range(0, 8):     
            v = self.robot.get_digital_in(i)
            inputs.append(v)
        print(inputs)
        return inputs
    
    def _startup_pins(self):
        self.robot.set_digital_out(0, 1)
        self.robot.set_digital_out(3, 1)
        self.robot.set_digital_out(2, 0)
        self.robot.set_digital_out(1, 0)
        # self.robot.set_digital_out(5, 1)  
        print(self.robot.get_scara_param())
        # self.safe_move_xyz(self.robot.x, self.robot.y, 0, SAFE_R, 50)
        # self.safe_move_xyz(50, 0, 0, -7.14, 50)
    
    def move_middle(self):

        self.safe_move_xyz(50, 0, 0, -7.14, 50)
        self.safe_move_xyz(50, 0, -100, -7.14, 50)
        self.robot.wait_stop()

    def move_up(self):
        self.safe_move_xyz(50, 0, 0, -7.14, 50)
        self.robot.wait_stop()
       
    def move_left(self):

        self.safe_move_xyz(50, 0, 0, -7.14, 50)
        self.safe_move_xyz(203.555, -316.625,0,SAFE_R, 50)
        self.safe_move_xyz(203.555, -316.625,-250,SAFE_R, 50)
        self.safe_move_xyz(203.555, -316.625,0,SAFE_R, 50)
        self.robot.wait_stop()

    def move_right(self):
        
        self.safe_move_xyz(50, 0, 0, -7.14, 50)
        self.safe_move_xyz(200.517525, 315.5884,0,SAFE_R, 50)
        self.safe_move_xyz(200.517525, 315.5884,-200,SAFE_R, 50)
        self.safe_move_xyz(200.517525, 315.5884,0,SAFE_R, 50)
        self.robot.wait_stop()

    def get_coor(self):
        return self.robot.x, self.robot.y, self.robot.z, self.robot.r
    
    def vacuum_on(self):
        self.send_command(self.SUCTION_ON, "Suction ON")
        time.sleep(0.2)

    def vacuum_off(self):
        self.send_command(self.SUCTION_RELEASE, "Suction RELEASE")
        time.sleep(0.2)


def main():
    """Main function to demonstrate usage."""
    robot = HitbotMove()
    
    # Set number of objects to process
    robot.set_iterations(4)
    
    # Process all available objects
    robot.run_all_objects()
    
    # Alternatively, process objects one by one
    # while robot.process_next_object():
    #     time.sleep(1)                             

if __name__ == "__main__":
    main()