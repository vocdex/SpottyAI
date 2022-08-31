import argparse
import grpc
import logging
import math
import os
import sys
import time
import threading

from bosdyn.api import geometry_pb2
from bosdyn.api import power_pb2
from bosdyn.api import robot_state_pb2
from bosdyn.api import basic_command_pb2
from bosdyn.api import image_pb2

import bosdyn.client
import bosdyn.client.channel
import bosdyn.client.util

from bosdyn.client.power import safe_power_off, PowerClient, power_on
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.api.graph_nav import map_pb2, map_processing_pb2, recording_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, LeaseWallet, ResourceAlreadyClaimedError
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient, build_image_request

import cv2
import numpy as np
from scipy import ndimage

import google.protobuf.timestamp_pb2
import numpy as np
import occupancyMapper.convert_mesh_ours as convert_mesh
from PIL import Image
import plyfile

View_thread_stop = False
#visited positions
visPosX = {}
visPosY = {}

def _stop_recording():
    """Stop or pause recording a map."""
    while True:
        try:
            status = recording_client.stop_recording()
            break
        except bosdyn.client.recording.NotReadyYetError as err:
            # It is possible that we are not finished recording yet due to
            # background processing. Try again every 1 second.
            time.sleep(1.0)
            continue
        except Exception as err:
            print("Stop recording failed: " + str(err))
            break



def _get_graphnav_origin():
    """ Returns seed_tform_body. """
    state = graph_nav_client.get_localization_state()
    gn_origin_tform_body = state.localization.seed_tform_body
    trafo = get_a_tform_b(robot.get_frame_tree_snapshot(), ODOM_FRAME_NAME, BODY_FRAME_NAME)
    pos = math_helpers.SE3Pose.from_proto(gn_origin_tform_body).position
    return trafo.transform_vec3(pos)

def _safe_curr_pos():
    """saves the robots current position in an array with all the other saved ones"""
    pos = _get_graphnav_origin()
    visPosX[len(visPosX)] = pos.x
    visPosY[len(visPosY)] = pos.y


ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()

id_list = []

def export_map(posX, posY):
    point_data = None
    current_anchors = {}
    map_processing_client.process_anchoring(
        params=map_processing_pb2.ProcessAnchoringRequest.Params(),
        modify_anchoring_on_server=True, stream_intermediate_results=False)
    graph = graph_nav_client.download_graph()
    #save anchors
    for anchor in graph.anchoring.anchors:
        current_anchors[anchor.id] = anchor
    #Download the waypoint snapshots from robot to the specified, local filepath location.
    num_waypoint_snapshots_downloaded = 0
    for waypoint in graph.waypoints:
        if len(waypoint.snapshot_id) == 0:
            continue
        try:
            waypoint_snapshot = graph_nav_client.download_waypoint_snapshot(waypoint.snapshot_id)
        except Exception:
            # Failure in downloading waypoint snapshot. Continue to next snapshot.
            print("Failed to download waypoint snapshot: " + waypoint.snapshot_id)
            continue
        #self._write_bytes(self._download_filepath + '/waypoint_snapshots','/' + waypoint.snapshot_id, waypoint_snapshot.SerializeToString())
        num_waypoint_snapshots_downloaded += 1
        #print("Downloaded {} of the total {} waypoint snapshots.".format(num_waypoint_snapshots_downloaded, len(waypoints)))
        cloud = waypoint_snapshot.point_cloud
        odom_tform_cloud = get_a_tform_b(cloud.source.transforms_snapshot, ODOM_FRAME_NAME, cloud.source.frame_name_sensor)
        waypoint_tform_odom = SE3Pose.from_obj(waypoint.waypoint_tform_ko)
        waypoint_tform_cloud = waypoint_tform_odom * odom_tform_cloud
        if waypoint.id not in current_anchors:
            raise Exception("{} not found in anchorings. Does the map have anchoring data?".format(waypoint_id))
        seed_tform_cloud = SE3Pose.from_obj(current_anchors[waypoint.id].seed_tform_waypoint) * waypoint_tform_cloud
        point_cloud_data = np.frombuffer(cloud.data, dtype=np.float32).reshape(int(cloud.num_points), 3)
        #cloud_data = get_point_cloud_data_in_seed_frame(current_waypoints, current_waypoint_snapshots, current_anchors, wp.id)
        if point_data is None:
            point_data = seed_tform_cloud.transform_cloud(point_cloud_data)
        else:
            point_data = np.concatenate((point_data, seed_tform_cloud.transform_cloud(point_cloud_data)))
    
    #print(point_data)
    #write_ply(point_data, "temp/room.ply")
    
    # starting from here its computing the occupancy map
    slice_config = {
        # 'offset':   0.5,  # vertical offset - percentage of z.max -z.min
        # 'interval': 0.05, # vertical interval - percentage of z.max -z.min

        'offset':   0.65,  # vertical offset - percentage of z.max -z.min
        'interval': 0.17, # vertical interval - percentage of z.max -z.min
    }

    ogm_config = {
        'mpp':             0.2, # meter per pixel ratio
        'margin':          10, # map margin
        'unexplored':      1, # value for unexplored pixels (.5:127 - 1.:255)
        'fill_neighbors':  False,
        'flip_vertically': True
    }
    #ply_data = plyfile.PlyData.read("temp/room.ply")
    Vx = point_data[:, 0] #ply_data['vertex'].data['x'] # f4 -> float32
    Vy = point_data[:, 1] #ply_data['vertex'].data['y'] # f4 -> float32
    Vz = point_data[:, 2] #ply_data['vertex'].data['z'] # f4 -> float32
    #ply_data, faces, [Vx, Vy, Vz, Vr, Vg, Vb, Va] = convert_mesh.load_ply_file( "temp/room.ply" )
    slice_idx = convert_mesh.slice_horizontal_vertices(Vz, slice_config)
    # generating the ogm
    ogm, mpp, OffPxX, OffPxY, currX, currY= convert_mesh.convert_2d_pointcloud_to_ogm(Vx,Vy, slice_idx, ogm_config, np.array(list(posX.items()))[:,1], np.array(list(posY.items()))[:,1])
    Image.fromarray(ogm).save('occupancyMapper/temp/room.png')
    return ogm, mpp, OffPxX, OffPxY, currX, currY

# Function to move robot to coordinates in its odometry frame
# Will return true once the robot has reached its position, false if the time limit is reached
# Arguments are: x position, y position, rotation in radians, a robot command client, the time limit for this move command
def walk_to_xyrot(x, y, rot, command_client, timeout_sec):
	walk_temp = RobotCommandBuilder.synchro_se2_trajectory_point_command(x, y, rot, 'odom')
	cmd_id = command_client.robot_command(walk_temp, end_time_secs=time.time() + timeout_sec)
	success = False
	start_time = time.time()
	end_time = start_time + timeout_sec
	now = time.time()
	while now < end_time:
		status = command_client.robot_command_feedback(cmd_id)
		status_out = status.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.status
		print(status_out)
		if status_out == 1:
			return True
		time.sleep(0.1)
		now = time.time()
	return false

def reset_image_client(robot):
    """Recreate the ImageClient from the robot object."""
    del robot.service_clients_by_name['image']
    del robot.channels_by_authority['api.spot.robot']
    return robot.ensure_client('image')
	
def image_to_opencv(image, auto_rotate=True):
    """Convert an image proto message to an openCV image."""
    num_channels = 1  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
        extension = ".png"
    else:
        dtype = np.uint8
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_channels = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_channels = 4
            dtype = np.uint16
        extension = ".jpg"

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    if auto_rotate:
        img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])
    return img, extension

def image_thread(name, image_client, robot):
	image_sources = ['frontleft_fisheye_image', 'frontright_fisheye_image', 'left_fisheye_image', 'right_fisheye_image', 'back_fisheye_image']
	requests = [
		build_image_request(source, quality_percent=50)
		for source in image_sources
	]
	for image_source in image_sources:
		cv2.namedWindow(image_source, cv2.WINDOW_NORMAL)
		cv2.setWindowProperty(image_source, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
	keystroke = None
	while keystroke != 113 and keystroke != 27 and View_thread_stop == False:
		try:
			images_future = image_client.get_image_async(requests, timeout=0.5)
			# print('Trying for image')
			while not images_future.done():
				keystroke = cv2.waitKey(25)
				if keystroke != -1:
					print(keystroke)
				if keystroke == 27 or keystroke == 113:
					sys.exit(1)
			# print('images done')
			images = images_future.result()
		except TimedOutError as time_err:
			if timeout_count_before_reset == 5:
				# To attempt to handle bad comms and continue the live image stream, try recreating the
				# image client after having an RPC timeout 5 times.
					# print('timeouterror')
					image_client = reset_image_client(robot)
					timeout_count_before_reset = 0
			else:
				timeout_count_before_reset += 1
		except Exception as err:
			continue
		for i in range(len(images)):
			# print('converting to opencv')
			image, _ = image_to_opencv(images[i], True)
			(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
			if len(corners) > 0:
				ids = ids.flatten()
				
				for (markerCorner, markerID) in zip(corners, ids):
					if markerID not in id_list:
						id_list.append(markerID)
						id_list.sort()
					
					corners = markerCorner.reshape((4, 2))
					(topLeft, topRight, bottomRight, bottomLeft) = corners
					
					topRight = (int(topRight[0]), int(topRight[1]))
					bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
					bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
					topLeft = (int(topLeft[0]), int(topLeft[1]))

					# draw the bounding box of the ArUCo detection
					cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
					cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
					cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
					cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

					# compute and draw the center (x, y)-coordinates of the ArUco
					# marker
					cX = int((topLeft[0] + bottomRight[0]) / 2.0)
					cY = int((topLeft[1] + bottomRight[1]) / 2.0)
					cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

					# draw the ArUco marker ID on the image
					cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
					#print("[INFO] ArUco marker ID: {}".format(markerID))

				# show the output image
			print(id_list)
			cv2.imshow(images[i].source.name, image)
		keystroke = cv2.waitKey(10)
	print('Image Thread terminated')

sdk = bosdyn.client.create_standard_sdk('walk_test')
robot = sdk.create_robot('192.168.80.3')
bosdyn.client.util.authenticate(robot)
robot.sync_with_directory()
robot.time_sync.wait_for_sync()

recording_client = robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
map_processing_client = robot.ensure_client(MapProcessingServiceClient.default_service_name)
image_client = robot.ensure_client(bosdyn.client.image.ImageClient.default_service_name)
	
#view_thread = threading.Thread(target=image_thread, args=(1,image_client,robot,))
#view_thread.start()
#recording_client = robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
#graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
#map_processing_client = robot.ensure_client(MapProcessingServiceClient.default_service_name)

lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
lease = lease_client.acquire()
lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(lease_client) 
	
robot.power_on(timeout_sec=20)
	
#command_client = robot.ensure_client(RobotCommandClient.default_service_name)
#blocking_stand(command_client, timeout_sec=10)
sit = RobotCommandBuilder.synchro_sit_command()
#clear graphNav graph on robot we want to start from begin when the robot starts

#walk_to_xyrot(0,0,0,command_client, 10)

_stop_recording()
graph_nav_client.clear_graph()
recording_client.start_recording() #start recording

while 1:
    _safe_curr_pos()
    ogm, mpp, OffPxX, OffPxY, currX, currY = export_map(visPosX,visPosY)
    #sizey = len(ogm[0,:])
    #sizex = len(ogm[:,0])
    #stepsize = 5
    #linsp = np.array((-2, -1, 0, 1, 2))
    #cx = 0
    #cy = 0
    ##test for valid directions
    #forward = True
    #right = True
    #left = True
    #back = True
    #for x in linsp:
    #    for y in linsp:
    #        cx = currX + stepsize + x
    #        cy = currY + y
    #        if cx >= sizex or cy >=sizey or cx < 0 or cy < 0 or ogm[x,y] == 0:
    #            forward = False
    #
    #if forward == False:
    #    right = True
    #    for x in linsp:
    #        for y in linsp:
    #            cx = currX + x
    #            cy = currY + stepsize + y
    #            if cx >= sizex or cy >=sizey or cx < 0 or cy < 0 or ogm[x,y] == 0:
    #                right = False
#
    #if forward == False and right == False:
    #    left = True
    #    for x in linsp:
    #        for y in linsp:
    #            cx = currX + x
    #            cy = currY - stepsize + y
    #            if cx >= sizex or cy >=sizey or cx < 0 or cy < 0 or ogm[x,y] == 0:
    #                left = False
#
    #if forward == False and right == False and left == False:
    #    back = True
    #    for x in linsp:
    #        for y in linsp:
    #            cx = currX - stepsize + x
    #            cy = currY + y
    #            if cx >= sizex or cy >=sizey or cx < 0 or cy < 0 or ogm[x,y] == 0:
    #                back = False
    #
    #if back == False:
    #    print("no way out")
    #    break
#
    #walk_to_xyrot(int(cx * mpp), int(cy * mpp), 0, command_client, 10)

#	for all points on map:
#		if point is not blocked and point is not visited:
#			walk_to_point
#			if walk_is_successful:
#				add_point_to_visited
#				update_map
#			else
#				add_position_to_visited
#				update_map
#			break
#		print('Done')
#		view_thread_stop = True


# walk_to_xyrot(0, 0, 0, command_client, 10)
#walk_to_xyrot(0, 0, 0, command_client, 10)
#_safe_curr_pos()
#walk_to_xyrot(1, 0, 0, command_client, 10)
#_safe_curr_pos()
#walk_to_xyrot(0, 0, 0, command_client, 10)
#_safe_curr_pos()

export_map(visPosX,visPosY)
_stop_recording()
#command_client.robot_command(sit)
robot.power_off()
lease_client.return_lease(lease)

	