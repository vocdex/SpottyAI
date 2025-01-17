# HOSTNAME = "192.168.80.3"
HOSTNAME = "192.168.182.93" # Via mobile hotspot
USERNAME = "user"
PASSWORD = "c037gcf6n93f"

 

def auto_authenticate(robot):
    robot.authenticate(USERNAME, PASSWORD, timeout=20)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()