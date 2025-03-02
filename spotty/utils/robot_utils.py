HOSTNAME="172.20.10.12" # Hotspot IP
USERNAME = "user"
PASSWORD = "c037gcf6n93f"

 

def auto_authenticate(robot):
    robot.authenticate(USERNAME, PASSWORD, timeout=20)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()