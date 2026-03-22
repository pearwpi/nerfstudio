from __future__ import annotations
import os
import time
import sys
sys.path.append("/home/pear_group/VizGoggles/nerfstudio")
sys.path.append("/home/pear_group/VizGoggles/Segment-and-Track-Anything")
from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, get_args)
import cv2
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import viser.transforms as vtf
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import ColormapOptions, Colormaps
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.viewer.vfh_obstacle_avoidance import ObstacleAvoidance
from nerfstudio.viewer.utils import CameraState, get_camera

# Dronekit and Pymavlink Modules
from dronekit import connect
import argparse
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal
import dronekit_sitl
from pymavlink import mavutil 
import math
import multiprocessing
import matplotlib.pyplot as plt
import threading

# Server Modules
from flask import Flask, send_file, Response
import logging
import json
import requests

# Misc Modules
import csv
import json

# navigation helpers
from .naviUtils import *

#Sam-Track Imports 
import os
import cv2
from PIL import Image
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from aot_tracker import _palette
import numpy as np
import torch
from scipy.ndimage import binary_dilation
import gc

# getting rid of log images from the terminal
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)

# sending images to the server
current_rgb_image_for_server       = None
current_depth_image_for_server     = None
current_position_cam_for_server    = None
current_orientation_cam_for_server = None

# Vicons Outer Bounds
VICON_BOUNDS_FILE = f"./nerfstudio/viewer/vicon_env/environment.txt" 

# Flags 
SAVE_IMAGES = True
OBSTACLE_AVOIDANCE = False
SCALE_FACTOR_FOR_Z_VEL = 1
# SCALE_FACTOR_FOR_Z_VEL = 1.5

###########################################################
# DEFINING SERVERS FOR CAM VISUALIZATION INSIDE THE SPLAT
###########################################################
# server for rgb images
@app.route('/rgb')
def rgb_image():
    global current_rgb_image_for_server
    _, img_encoded = cv2.imencode('.png', current_rgb_image_for_server)
    return Response(img_encoded.tostring(), mimetype='image/png')

# server for depth images
@app.route('/depth')
def depth_image():
    global current_depth_image_for_server
    _, img_encoded = cv2.imencode('.png', current_depth_image_for_server)
    return Response(img_encoded.tostring(), mimetype='image/png')

@app.route('/current_position')
def current_position():
    global current_position_cam_for_server
    if current_position_cam_for_server is not None:
        # Convert the NumPy array to a list of lists
        position_data = {
            'position_matrix': current_position_cam_for_server.tolist()
        }
        return json.dumps(position_data), 200, {'Content-Type': 'application/json'}
    else:
        return "No position data available", 404

@app.route('/current_orientation')
def current_orientation():
    global current_orientation_cam_for_server
    if current_orientation_cam_for_server is not None:
        # Convert the NumPy array to a list of lists
        orientation_data = {
            'orientation_matrix': current_orientation_cam_for_server.tolist()
        }
        return json.dumps(orientation_data), 200, {'Content-Type': 'application/json'}
    else:
        return "No orientation data available", 404


# run flask server
def run_flask():
    app.run(host='0.0.0.0', port=8008, threaded=True)

######################
# CUSTOM VIEWER CLASS
######################
@decorate_all([check_main_thread])
class Custom_Viewer(object):
    """
    Custom Viewer class for rendering images and depth maps using NeRFStudio.
    """
    config: TrainerConfig
    pipeline: Pipeline
    
    def __init__(self, config: TrainerConfig, pipeline: Pipeline, vehicle = None, save_imgs: bool = SAVE_IMAGES, obstacle_avoidance: bool = OBSTACLE_AVOIDANCE, environment_file: str = VICON_BOUNDS_FILE):
        super().__init__()
        self.config = config
        self.pipeline = pipeline
        self.output_keys = {}
        
        self.obstacle_avoidance = obstacle_avoidance
        self.save_imgs = save_imgs    
        self.vehicle = vehicle    
        
        self.model = self.pipeline.model
        self.background_color = torch.tensor([0.1490, 0.1647, 0.2157], device=self.model.device)
        self.scale_ratio = 1
        self.max_res = 640
        self.image_height = 480
        self.image_width = 640
        self.depth_res = 640
        self.fov = 1.3089969389957472
        self.aspect_ratio = 1.774976538533895
        self.colormap_options_rgb = ColormapOptions(colormap='default', normalize=True, colormap_min=0.0, colormap_max=1.0, invert=False)
        self.colormap_options_depth = ColormapOptions(colormap='gray', normalize=True, colormap_min=0.0, colormap_max=1.0, invert=False)
        self.i = 0
        
        # calculate focal length based on image height and field of view   
        pp_h = self.image_height / 2.0
        self.focal_length = pp_h / np.tan(self.fov / 2.0)
        self.fx = self.focal_length
        self.fy = self.focal_length
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
                
        # Initialize your desired obstacle avoidance planner
        self.vfh_planner = ObstacleAvoidance(height=self.image_height, width=self.image_width, focal_length=self.focal_length)
        
        # for wire tracking
#    tensor([[-0.9966, -0.0819,  0.0049, -0.0621],
#         [-0.0049,  0.1193,  0.9928,  0.8313],
#         [-0.0819,  0.9895, -0.1193, -0.9520]]
        self.init_position    = np.array([1.7079e-03, 5.1221e-01, -1.1945e+00])
        self.init_orientation = np.array([[ -0.9966, -0.0819,  0.0049],
                                          [ -0.0049,  0.1193,  0.9928],
                                          [ -0.0819,  0.9895, -0.1193]])
        
        # self.init_position = np.array([5.9634e-01, -3.3720e-01, 3.2203e-03 ])
        # self.init_orientation = np.array([[-9.9533e-01,  5.9834e-04, -9.6545e-02],
        #                                   [-9.6546e-02, -6.1685e-03,  9.9531e-01],
        #                                   [5.5511e-17,  9.9998e-01,  6.1974e-03]])
    
        # self.init_position = np.array([7.5731e-01, -2.9925e+00, -1.3526e+00])
        # self.init_orientation = np.array([[ 9.5918e-01, -4.7374e-02,  2.7879e-01],
        #                           [ 2.8279e-01,  1.6069e-01, -9.4563e-01],
        #                         [ 5.5511e-17,  9.8587e-01,  1.6752e-01]])
        
        # creating the directories and sub-directories outputs
        self.experiment_directory = "./RaytheonTest"
        os.makedirs(self.experiment_directory, exist_ok=True)
        self.create_output_directory()
        self.rgb_directory = os.path.join(self.output_directory, "rgb")
        self.depth_directory = os.path.join(self.output_directory, "depth")
        self.gtpose_directory = os.path.join(self.output_directory, "gt_poses")
        self.wire_mask_directory = os.path.join(self.output_directory, "wire_mask")
        self.rgb_for_mask_directory = os.path.join(self.output_directory, "rgb for_mask")
        self.depth_for_mask_directory = os.path.join(self.output_directory, "depth for_mask")
        self.skeleton_mask_directory = os.path.join(self.output_directory, "skeleton_mask_directory")
        self.point_clouds_directory = os.path.join(self.output_directory, "point_cloud_directory")
        self.control_command_directory = os.path.join(self.output_directory, "control_command_directory")
        self.depth_for_mask_filtered_directory = os.path.join(self.output_directory, "depth_for_mask_filtered_directory")
        
        
        os.makedirs(self.rgb_directory, exist_ok=True)
        os.makedirs(self.depth_directory, exist_ok=True)
        os.makedirs(self.gtpose_directory, exist_ok=True)
        os.makedirs(self.wire_mask_directory, exist_ok=True)
        os.makedirs(self.rgb_for_mask_directory, exist_ok=True)
        os.makedirs(self.depth_for_mask_directory, exist_ok=True)
        os.makedirs(self.skeleton_mask_directory, exist_ok=True)
        os.makedirs(self.point_clouds_directory, exist_ok=True)
        os.makedirs(self.control_command_directory, exist_ok=True)
        os.makedirs(self.depth_for_mask_filtered_directory, exist_ok=True)
        self.csv_file = os.path.join(self.gtpose_directory, 'poses_in_camera_frame.csv')
        self.initialize_csv()
        
        # misc
        self.land = False
        self.lock = threading.Lock()
        self.counter = 0 
        self.image_id = 0
        self.stop_saving_images = threading.Event()
        self.environment_file = environment_file
        self.environment_bounds = self.read_environment_file(self.environment_file)
        self.FIRST_FRAME_WIRE = True
        
        self.current_waypoint_idx = 0
        self.guided_first_time = True
       
        self.origin_pose = np.array([None,None,None,None,None,None])
        if np.all(self.origin_pose == None):
            self.origin_pose = self.get_latest_pose()

        #tracking params
        segtracker_args = {'sam_gap': 9999, 'min_area': 200, 'max_obj_num': 255, 'min_new_obj_iou': 0.8}
        sam_args = {'sam_checkpoint': '/home/pear_group/VizGoggles/Segment-and-Track-Anything/ckpt/sam_vit_b_01ec64.pth', 'model_type': 'vit_b', 'generator_args': {'points_per_side': 16, 'pred_iou_thresh': 0.8, 'stability_score_thresh': 0.9, 'crop_n_layers': 1, 'crop_n_points_downscale_factor': 2, 'min_mask_region_area': 200}, 'gpu_id': 0}
        aot_args = {'phase': 'PRE_YTB_DAV', 'model': 'r50_deaotl', 'model_path': '/home/pear_group/VizGoggles/Segment-and-Track-Anything/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth', 'long_term_mem_gap': 9999, 'max_len_long_term': 9999, 'gpu_id': 0}
        self.segtracker = SegTracker(segtracker_args,sam_args,aot_args)
        torch.cuda.empty_cache()
        gc.collect()
        self.sam_gap = segtracker_args['sam_gap']
        self.wire_caption = "wire"
        self.box_threshold = 0.40
        self.text_threshold = 0.25
        self.track_id = 0
    #########################
    # GENERAL UTILS METHODS
    #########################
    def read_environment_file(self, file_path):
        """
        Reads the environmnet.txt file and extracts the boundaries. 
        Assumes the file format is:
        min_x: value
        max_x: value
        min_y: value
        max_y: value
        min_z: value
        max_z: value
        """
        bounds = {}
        
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                bounds[key.strip()] = float(value.strip())
        
        return bounds
    
    def monitor_bounds(self, current_local_position_for_monitoring):
        """
        Check if the drone is within the bounds of the vicon space
        """
        if (current_local_position_for_monitoring[0] < self.environment_bounds['min_x'] or
            current_local_position_for_monitoring[0] > self.environment_bounds['max_x'] or
            current_local_position_for_monitoring[1] < self.environment_bounds['min_y'] or
            current_local_position_for_monitoring[1] > self.environment_bounds['max_y'] or
            current_local_position_for_monitoring[2] < self.environment_bounds['min_z'] or
            current_local_position_for_monitoring[2] > self.environment_bounds['max_z']):
            print("Drone out of bounds! Initiating return to land.")
            time.sleep(0.001)
            self.land = True
            self.return_to_land()
   
        
    def create_output_directory(self):
        """
        Creating output directory for saving images
        """
        subdirs = [d for d in os.listdir(self.experiment_directory) if os.path.isdir(os.path.join(self.experiment_directory, d))]
        output_dirs = [d for d in subdirs if d.startswith('output') and d[6:].isdigit()]
        if output_dirs:
            highest_index = max([int(d[6:]) for d in output_dirs])
        else:
            highest_index = 0
        output_dir_name = f'output{highest_index + 1}'
        self.output_directory = os.path.join(self.experiment_directory, output_dir_name)
        os.makedirs(self.output_directory)
        print(f'Created new output directory: {self.output_directory}')

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Converts Euler Angles to Quaternion
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qw, qx, qy, qz]
    
    def quaternion_multiply(self, quaternion1, quaternion0):
        """
        Multiplies 2 quaternions
        """
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    
    def initialize_csv(self):
        """
        Initialize the CSV file to save the poses.
        """
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow([
                'Image_Index', 'x', 'y', 'z',
                'r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33'
            ])

    def write_pose_to_csv(self, img_index: int, position: np.ndarray, orientation: np.ndarray):
        """
        Write the position and orientation (rotation matrix) to the CSV file.
        """
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [img_index] + list(position) + orientation.flatten().tolist()
            writer.writerow(row)

    def rotation_matrix_from_euler(self, roll, pitch, yaw):
        """
        make rotation matrix from rol, pitch, yaw
        """
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])
        return R
    
    def get_latest_pose(self):
        try:
            response = requests.get("http://localhost:8000/get_pose")
            data = response.json()
            x = data["x"]
            y = data["y"]
            z = data["z"]
            roll = data["roll"]
            pitch = data["pitch"]
            yaw = data["yaw"]
            return np.array([x, y, z, roll, pitch, yaw])
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None
    
    def convert_to_ned(self, current_pose, origin_pose):
        """
        convert from NWU to NED frame 
        """    
        x =  (current_pose[0] - origin_pose[0]) * 0.001 
        y = -(current_pose[1] - origin_pose[1]) * 0.001 
        z = -(current_pose[2] - origin_pose[2]) * 0.001
        roll = current_pose[3] - origin_pose[3]  
        pitch = -(current_pose[4] - origin_pose[4])  
        yaw = -(current_pose[5] - origin_pose[5])
        
        return np.array([x,y,z]), np.array([roll, pitch, yaw])

    ######################
    # DRONEKIT FUNCTIONS
    ######################    
    def goto_position_target_local_ned(self, north, east, down):
        """
        Send SET_POSITION_TARGET_LOCAL_NED command to request the vehicle fly to a specified
        location in the North, East, Down frame.
        """
        # msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
        #     0,       
        #     0, 0,    
        #     mavutil.mavlink.MAV_FRAME_LOCAL_NED, 
        #     0b0000111111111000, # type_mask (only positions enabled)
        #     north, east, down,
        #     0.0,0.0, 0.0, # x, y, z velocity in m/s  (not used)
        #     0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        #     0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
        # # send command to vehicle
        # self.vehicle.send_mavlink(msg)

        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,       
            0, 0,    
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, 
            0b0000110111111000, # type_mask (only positions enabled)
            north, east, down,
            0.0,0.0, 0.0, # x, y, z velocity in m/s  (not used)
            0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
            0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
        # send command to vehicle
        self.vehicle.send_mavlink(msg)
        
    def init_vehicle(self): # {NOTE: Vehicle Initialization takes place inside the run_viewer_custom_multithreading.py script and not in this script}
        """
        Connects with the Vehicle over Pear Drone wifi 
        The positions and orientations are in the vicon frame of reference which is front x, left y, up z
        """
        # Logging vehicle info and connecting to the vehicle using connection string
        connection_string = 'udpin:0.0.0.0:14551'
        print('Connecting to vehicle on: %s' % connection_string)
        self.vehicle = connect(connection_string, wait_ready=True, rate=90)
        self.vehicle.parameters['SR0_POSITION'] = 90 
        self.vehicle.parameters['SR1_POSITION'] = 90
        print("Connected.")
        print("Init Mode: %s" % self.vehicle.mode.name)
        
    def return_to_land(self):
        """
        Changes the mode of the robot to Return to Land Mode
        """
        with self.lock:
            if self.vehicle.mode.name != "LAND":
                print("Changing mode to LAND")
                self.vehicle.mode = VehicleMode("LAND")
                print("Mode changed to LAND")
            
    def get_current_ned_vel(self):
        """
        Get current velocity for logging
        """
        return self.vehicle.velocity
    
    def send_ned_velocity(self, velocity_x, velocity_y, velocity_z, duration=1):
        """
        Sends velocity commands to the drone at the rate of 100 Hz
        """
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,                                      
            0, 0,                                   
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,    
            0b0000111111000111,                     
            0, 0, 0,                                
            velocity_x, velocity_y, velocity_z,     
            0, 0, 0,                                
            0, 0)                                    

        # send command to vehicle on 1000 Hz cycle
        self.vehicle.send_mavlink(msg)
        time.sleep(0.01)
        
    def condition_yaw(self, heading, relative=False):
        """
        Gives yaw to the robot. 
        """
        if relative:
            is_relative=1 #yaw relative to direction of travel
        else:
            is_relative=0 #yaw is an absolute angle
        # create the CONDITION_YAW command using command_long_encode()
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
            0, #confirmation
            heading,    # param 1, yaw in degrees
            0,    # param 2, yaw speed deg/s
            0,          # param 3, direction -1 ccw, 1 cw
            is_relative, # param 4, relative offset 1, absolute angle 0
            0, 0, 0)    # param 5 ~ 7 not used
        # send command to vehicle
        self.vehicle.send_mavlink(msg)
        # time.sleep(1)

    #########################
    # Nerfstudio Functions
    #########################
    def get_camera_state(self, rot_mat, pos):
        """
        Gets the state of the camera
        """
        rot_adjustment = np.eye(3)
        r = self.rotation_matrix_from_euler(np.radians(0), 0, 0)
        R = rot_adjustment @ rot_mat
        R = torch.tensor(r @ R)   
        pos = torch.tensor([pos[0], pos[1], pos[2]], dtype=torch.float64) / self.scale_ratio
        c2w = torch.concatenate([R, pos[:, None]], dim=1)
        camera_state = CameraState(fov=self.fov, aspect=self.aspect_ratio, c2w=c2w, camera_type=CameraType.PERSPECTIVE)
        return camera_state
    
    def _render_img(self, camera_state: CameraState):
        """
        Renders the RGB and depth images at the current camera frame
        """
        obb = None
        camera = get_camera(camera_state, self.image_height, self.image_width)
        camera = camera.to(self.model.device)
        self.model.set_background(self.background_color)
        self.model.eval()
        outputs = self.model.get_outputs_for_camera(camera, obb_box=obb)
        desired_depth_pixels = self.depth_res
        current_depth_pixels = outputs["depth"].shape[0] * outputs["depth"].shape[1]
        scale = min(desired_depth_pixels / current_depth_pixels, 1.0)
        
        return outputs
    
    
    def process_outputs(self, outputs):
        """
        Processes outputs to compute velocities based on images
        """
        rgb = outputs['rgb']
        depth = outputs["depth"]
        rgb_image = colormaps.apply_colormap(image=rgb, colormap_options=self.colormap_options_rgb)
        rgb_image = (rgb_image * 255).type(torch.uint8)
        rgb_image = rgb_image.cpu().numpy()
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        depth_image = colormaps.apply_colormap(image=depth, colormap_options=self.colormap_options_depth)
        depth_image = (depth_image * 255).type(torch.uint8)
        depth_image = depth_image.cpu().numpy()
        
        return rgb_image, depth_image
        
    def current_images(self, outputs):
        """
        Returns RGB and Depth Image Outputs
        """
        rgb = outputs['rgb']
        depth = outputs["depth"]
        rgb_image = colormaps.apply_colormap(image=rgb, colormap_options=self.colormap_options_rgb)
        rgb_image = (rgb_image * 255).type(torch.uint8)
        rgb_image = rgb_image.cpu().numpy()
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        depth_image = colormaps.apply_colormap(image=depth, colormap_options=self.colormap_options_depth)
        depth_image = (depth_image * 255).type(torch.uint8)
        depth_image = depth_image.cpu().numpy()
        
        return rgb_image, depth_image

    #######################
    # EXECUTION FUNCTIONS
    #######################  
    def SegTracker_add_first_frame(self, origin_frame, predicted_mask):
        
        with torch.cuda.amp.autocast():
            # Reset the first frame's mask
            frame_idx = 0
            self.segtracker.restart_tracker()
            self.segtracker.add_reference(origin_frame, predicted_mask, frame_idx)
            self.segtracker.first_frame_mask = predicted_mask

    
    def track(self, current_image):
        
        frame = cv2.cvtColor(current_image,cv2.COLOR_BGR2RGB)
        
        with torch.cuda.amp.autocast():
            if self.track_id == 0:
                pred_mask, _= self.segtracker.detect_and_seg(frame, self.wire_caption, self.box_threshold, self.text_threshold)
                # pred_mask_electric, _= self.segtracker.detect_and_seg(frame, self.electric_caption, self.box_threshold, self.text_threshold)
                # pred_mask = pred_mask_wire + pred_mask_electric
                self.SegTracker_add_first_frame(frame, pred_mask)
                torch.cuda.empty_cache()
                gc.collect()
            
            elif (self.track_id % self.sam_gap) == 0:
                seg_mask = self.segtracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = self.segtracker.track(frame)
                new_obj_mask = self.segtracker.find_new_objs(track_mask,seg_mask)
                pred_mask = track_mask + new_obj_mask
                self.segtracker.add_reference(frame, pred_mask)
            
            else:
                pred_mask = self.segtracker.track(frame,update_memory=True)
                torch.cuda.empty_cache()
                gc.collect()
            self.track_id+=1
        return pred_mask
            
              
    def save_images(self):
        
        global current_rgb_image_for_server, current_depth_image_for_server
        global current_position_cam_for_server, current_orientation_cam_for_server
        
        while not self.stop_saving_images.is_set():
            
            # Getting local NED pose for image saving
            # current_local_position_for_saving_ned, current_local_orientation_for_saving_ned = self.get_current_local_pose()
            current_pose_for_saving = self.get_latest_pose()
            current_local_position_for_saving_ned, current_local_orientation_for_saving_ned = self.convert_to_ned(current_pose_for_saving, self.origin_pose) 
            # Computing position update within splat for drone control {NOTE: This is the current position the drone. Called update because we'll add it to the Origin Position}
            position_update_cam_for_saving = self.init_orientation @ np.array([[current_local_position_for_saving_ned[1]],
                                                                               [-current_local_position_for_saving_ned[2]*SCALE_FACTOR_FOR_Z_VEL],
                                                                               [-current_local_position_for_saving_ned[0]]])
            
            # Updating the position in the splat for drone control loop
            current_position_cam_for_saving = [self.init_position[0] + position_update_cam_for_saving[0, 0],
                                               self.init_position[1] + position_update_cam_for_saving[1, 0],
                                               self.init_position[2] + position_update_cam_for_saving[2, 0]]
            
            # Updated Orientation in the nerfstudio cam frame
            # r_for_saving = Rotation.from_euler('xyz', [current_local_orientation_for_saving_ned.pitch, -current_local_orientation_for_saving_ned.yaw, -current_local_orientation_for_saving_ned.roll], degrees=False).as_matrix()
            r_for_saving = self.rotation_matrix_from_euler(current_local_orientation_for_saving_ned[1], -current_local_orientation_for_saving_ned[2], -current_local_orientation_for_saving_ned[0])
            
            # Current Orientation with respect to the origin frame in nerfstudio
            current_orientation_cam_for_saving = self.init_orientation @ r_for_saving
            
            # save current orientation and position to csv
            self.write_pose_to_csv(self.image_id, current_position_cam_for_saving, current_orientation_cam_for_saving)
            
            # Getting camera state, rendering RGB and depth images at that camera state 
            camera_state_for_saving = self.get_camera_state(current_orientation_cam_for_saving, current_position_cam_for_saving)
            outputs_for_saving = self._render_img(camera_state_for_saving)
            current_rgb_for_saving, current_depth_for_saving = self.current_images(outputs_for_saving)
            
            # # Putting RGB and depth images into the queue for the server to access
            current_rgb_image_for_server = current_rgb_for_saving
            current_depth_image_for_server = current_depth_for_saving
            current_position_cam_for_server = current_orientation_cam_for_saving
            current_orientation_cam_for_server = current_orientation_cam_for_saving
            
            # Saving images locally
            cv2.imwrite(os.path.join(self.rgb_directory, f"{self.image_id:04d}.png"), current_rgb_for_saving)
            cv2.imwrite(os.path.join(self.depth_directory, f"{self.image_id:04d}.png"), current_depth_for_saving)
            
            with open(self.gtpose_directory + "/log_debug.txt", "a") as file:
                print(time.time(), current_position_cam_for_saving, file=file)
            
            # Updating index for next iteration of the loop
            self.image_id += 1
            
            # Loop rate of saving images
            # time.sleep(0.05)

    def control_loop(self):
        
        # initializing empty pose and timestamp list for logging
        time_list  = []
        north_list = []
        down_list  = []
        east_list  = []
        vx_list = []
        vy_list = []
        vz_list = []
        last_save_time = time.time()
        
        starting_time = time.time()

        import pdb
        
        def getWorld2View2(c2w):
            # This code is taken from nerfstudio splatfacto rendering code. They use different conventions to mess with the peace of end users
            R = c2w[:3,:3]  # 3 x 3
            T = c2w[:3,3].reshape(3,1)  # 3 x 1
            # T = T.reshape(3,1)  # 3 x 1
            
            # flip the z and y axes to align with gsplat conventions
            R_edit = np.diag([1, -1, -1])
            R = R @ R_edit
            # analytic matrix inverse to get world2camera matrix
            R_inv = R.T
            T_inv = -R_inv @ T
            viewmat = np.eye(4)
            viewmat[:3, :3] = R_inv
            viewmat[:3, 3:4] = T_inv
            return np.float32(viewmat)
        
        while not self.stop_saving_images.is_set():
            # FOR DEBUGGING ADD LAND TO THE BELOW LIST. DONT CHANGE WHILE NOT SELF.LAND
            if self.vehicle.mode.name in ["LOITER", "GUIDED"] and time.time()>starting_time+5:
                while not self.land:
                    try:                        
                        # Extracting the current pose from vehicle which is in the NED Frame for drone control loop
                        # current_local_position_for_control_ned, current_local_orientation_for_control_ned = self.get_current_local_pose()
                        current_pose_for_control = self.get_latest_pose()
                        current_local_position_for_control_ned, current_local_orientation_for_control_ned = self.convert_to_ned(current_pose_for_control, self.origin_pose)
                        
                        # check if the drone is out of bounds in the vicon space
                        self.monitor_bounds(current_local_position_for_control_ned)
                    
                        
                        # if self.counter%100 == 0:
                        #     print(f"\nNorth: {current_local_position_for_control_ned[0], }\n")
                        #     print(f"\nEast: {current_local_position_for_control_ned[1], }\n")
                        #     print(f"\nDown: {current_local_position_for_control_ned[2], }\n")
                                                                        
                        # Computing position update within splat for drone control {NOTE: This is the current position the drone. Called update because we'll add it to the Origin Position}
                        position_update_cam_for_control = self.init_orientation @ np.array([[current_local_position_for_control_ned[1]],
                                                                                            [-current_local_position_for_control_ned[2]],
                                                                                            [-current_local_position_for_control_ned[0]]])
                        
                        # Updating the position in the splat for drone control loop
                        current_position_cam_for_control = [self.init_position[0] + position_update_cam_for_control[0, 0],
                                                            self.init_position[1] + position_update_cam_for_control[1, 0],
                                                            self.init_position[2] + position_update_cam_for_control[2, 0]] 
                        
                        # Updated Orientation in the nerfstudio cam frame
                        # r_for_control = Rotation.from_euler('xyz', [current_local_orientation_for_control_ned.pitch, -current_local_orientation_for_control_ned.yaw, -current_local_orientation_for_control_ned.roll], degrees=True).as_matrix()
                        r_for_control = self.rotation_matrix_from_euler(current_local_orientation_for_control_ned[1],
                                                                        -current_local_orientation_for_control_ned[2],
                                                                        -current_local_orientation_for_control_ned[0])

                        # Current Orientation with respect to the origin frame in nerfstudio
                        current_orientation_cam_for_control = self.init_orientation @ r_for_control
                        
                        # Rendering images based on the updated position
                        camera_state_for_control = self.get_camera_state(current_orientation_cam_for_control, current_position_cam_for_control)
                        outputs_for_control = self._render_img(camera_state_for_control)
                        rgb_image, depth_image = self.process_outputs(outputs_for_control)
                        
                        # Sending velocity commands to the drone for debugging
                        # TODO remove the land and set to guided
                        if self.vehicle.mode.name == "GUIDED":
                        # if self.vehicle.mode.name == "LAND":
                            #start tracking from here
                            if self.track_id == 0:
                                # first_ned = [0.697610, -0.578423 ,-0.893467]
                                # self.goto_position_target_local_ned(first_ned[0],
                                #                                     first_ned[1],
                                #                                     first_ned[2])
                                # self.condition_yaw(0, False)
                                # time.sleep(10)
                                # print("Reached First NED")
                                starting_ned = current_local_position_for_control_ned.copy()
                            pred_mask = self.track(rgb_image)
                            cv2.imwrite(os.path.join(self.wire_mask_directory, str(self.track_id)+".png"), np.uint8(pred_mask*255))
                            cv2.imwrite(os.path.join(self.rgb_for_mask_directory, str(self.track_id)+".png"), np.uint8(rgb_image))
                            
                            depth_mm = (depth_image * 1000).astype(np.uint16) 
                            cv2.imwrite(os.path.join(self.depth_for_mask_directory, str(self.track_id)+".png"), depth_mm)
                            
                            transformToSplat = np.linalg.inv(getWorld2View2(camera_state_for_control.c2w.numpy()))
                            
                            current_pc, current_center_pc, skeleton, filtered_depth = getPointsFromImage(pred_mask, [self.cx, self.cy, self.fx, self.fy], outputs_for_control['depth'], transformToSplat, self.init_position, self.init_orientation)
                            
                            filtered_depth_mm = (filtered_depth * 1000).astype(np.uint16) 
                            cv2.imwrite(os.path.join(self.depth_for_mask_filtered_directory, str(self.track_id)+".png"), filtered_depth_mm)
                            cv2.imwrite(os.path.join(self.skeleton_mask_directory, str(self.track_id)+".png"), np.uint8(skeleton*255))
                            
                            if self.FIRST_FRAME_WIRE:
                                # init the storage 
                                self.seen_pc = current_center_pc.copy()
                                self.trajectory_pc = current_pc.copy()
                                self.FIRST_FRAME_WIRE = False

                            # add the points that are already seen
                            if len(self.trajectory_pc) != 0 and len(current_pc) != 0:
                                self.trajectory_pc = merge_point_clouds(self.trajectory_pc, current_pc, distance_threshold = DIST_THRES)
                            elif len(current_pc) !=0 and len(self.trajectory_pc) == 0:
                                self.trajectory_pc = current_pc.copy()
                            
                            if len(self.seen_pc) != 0 and len(current_center_pc) != 0:
                                self.seen_pc = merge_point_clouds(self.seen_pc, current_center_pc, distance_threshold = DIST_THRES)
                            elif len(current_center_pc) != 0 and len(self.seen_pc) == 0:
                                self.seen_pc = current_center_pc.copy()
                            
                            if len(self.trajectory_pc) !=0:
                                np.savetxt(os.path.join(self.control_command_directory, str(self.track_id) + "all_pc"  + ".txt"), self.trajectory_pc, fmt='%f', delimiter=',')
                            
                            if len(self.seen_pc) !=0:
                                np.savetxt(os.path.join(self.control_command_directory, str(self.track_id) + "seen_pc" + ".txt"), self.seen_pc,       fmt='%f', delimiter=',')
                            
                            # The planner needs to take us to the closest unseen point from the current location
                            # TODO - what if the initial pc are empty
                            if len(self.trajectory_pc) !=0:
                                if len(self.seen_pc) == 0:
                                    unseenPoints = self.trajectory_pc.copy()
                                else:
                                    unseenPoints = set_difference_pc(self.trajectory_pc, self.seen_pc)

                                # next waypoint is
                                # TODO set the current position as a 1d numpy array or python list
                                curr_pos_ned = current_local_position_for_control_ned.copy()
                                
                                np.savetxt(os.path.join(self.control_command_directory, str(self.track_id) + "curr_pos_ned" + ".txt"), curr_pos_ned,       fmt='%f', delimiter=',')

                                # Outlier issues
                                
                                norms = np.linalg.norm(unseenPoints, axis=1)
                                
                                unseenPointsFiltered = unseenPoints[norms<5].copy()
                                print(len(unseenPointsFiltered))
                                if len(unseenPointsFiltered):
                                    next_traj_point, min_dist, min_idx  = find_closest_point(unseenPointsFiltered, curr_pos_ned)
                                    
                                    next_wp = next_traj_point.copy()
                                    next_wp[0] = next_wp[0] - 0.2
                                    if np.linalg.norm(next_wp - curr_pos_ned) < 0.2:
                                        self.seen_pc = np.vstack((self.seen_pc, next_traj_point.reshape(1,3)))
                                else:
                    
                                    next_wp = curr_pos_ned
                                print("Waypoint:", next_wp)
                                print("Current NED:", curr_pos_ned)
                                # self.goto_position_target_local_ned(next_wp[0],
                                #                                     next_wp[1],
                                #                                     next_wp[2])
                                # self.condition_yaw(0, False)
                                
                                self.goto_position_target_local_ned(next_wp[0],
                                                                    next_wp[1],
                                                                    next_wp[2])
                                self.condition_yaw(0, False)
                                time.sleep(0.1)
                                np.savetxt(os.path.join(self.control_command_directory, str(self.track_id) + "next_wp" + ".txt"), next_wp,fmt='%f', delimiter=',')
                            else:
                                continue
                    # Stop drone control after keyboard interrupt                
                    except KeyboardInterrupt:
                        self.stop_saving_images.set()
                        break
        time.sleep(0.1)
        
        # Land the drone, and close all the processes. 
        print("Landing")
        self.return_to_land()

    def execute(self):
        # Start Flask server process {Thread 1}
        flask_process = threading.Thread(target=run_flask)
        flask_process.start()

        # Initialize vehicle in the main process
        if self.vehicle is None:
            self.init_vehicle()

        # Start image saving in the main process if it involves vehicle operations {Thread 2}
        save_image_thread = threading.Thread(target=self.save_images)
        save_image_thread.start()

        # Start control loop in the main process if it involves vehicle operations {Thread 3}
        control_loop_thread = threading.Thread(target=self.control_loop)
        control_loop_thread.start()
        
        try:
            flask_process.join()
            save_image_thread.join()
            control_loop_thread.join()
        except KeyboardInterrupt:
            print("Interrupted! Attempting to land the drone...")
            self.stop_saving_images.set()
            self.return_to_land()
        finally:
            print("Execution stopped, ensuring the drone has landed.")
            # Double-checking to ensure the drone is landed
            if self.vehicle.mode.name != "LAND":
                self.return_to_land()
            print("Landing completed. Cleaning up resources.")
