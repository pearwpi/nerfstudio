from __future__ import annotations
import os
import time
import sys
sys.path.append("/home/pear_group/VizGoggles/nerfstudio")
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
# from nerfstudio.viewer.vfh_obstacle_avoidance import ObstacleAvoidance
from nerfstudio.viewer.utils import CameraState, get_camera

# Misc Modules
import csv
import json
import pandas as pd


@decorate_all([check_main_thread])
class Custom_Viewer(object):
    """
    Custom Viewer class for rendering images and depth maps using NeRFStudio.
    """
    config: TrainerConfig
    pipeline: Pipeline
    
    def __init__(self, config: TrainerConfig, pipeline: Pipeline):
        super().__init__()
        self.config = config
        self.pipeline = pipeline
        self.output_keys = {}
        
        # self.obstacle_avoidance = obstacle_avoidance
        # self.save_imgs = save_imgs    
        # self.vehicle = vehicle   
        
        self.model = self.pipeline.model
        self.background_color = torch.tensor([0.1490, 0.1647, 0.2157], device=self.model.device)
        self.scale_ratio = 1
        self.max_res = 640
        self.image_height = 480
        self.image_width = 640
        self.depth_res = 640
        self.fov = 1.30
        self.aspect_ratio = self.image_width/self.image_height
        self.colormap_options_rgb = ColormapOptions(colormap='default', normalize=True, colormap_min=0.0, colormap_max=1.0, invert=False)
        self.colormap_options_depth = ColormapOptions(colormap='gray', normalize=True, colormap_min=0.0, colormap_max=1.0, invert=False)
        self.i = 0
        self.save_path = "data/output136/depth_all"
        
        # calculate focal length based on image height and field of view   
        pp_h = self.image_height / 2.0
        self.focal_length = pp_h / np.tan(self.fov / 2.0)
        
        # camera intrinsics
        self.fx = self.focal_length
        self.fy = self.focal_length
        self.cx = self.image_height / 2
        self.cy = self.image_width / 2
        print(self.fx, self.fy, self.cx, self.cy)
        # Initialize your desired obstacle avoidance planner
        
        # for washburn obstacle course {Forward Facing Camera}
        # self.init_position    = np.array([-2.7377e+00, -3.5406e+00, -2.0152e-01])

        #For forest scene
        # tensor([[ 0.9086,  0.0288, -0.4166, -0.5314],
        # [-0.4176,  0.0627, -0.9065, -3.0209],
        # [ 0.0000,  0.9976,  0.0690,  0.8309]], dtype=torch.float64)
    
        # creating the directories and sub-directories outputs
    
        # set origin as where you start
        # self.origin_pose = np.array([None,None,None,None,None,None])
        # if np.all(self.origin_pose == None):
        #     self.origin_pose = self.get_latest_pose()
    
    #########################
    # GENERAL UTILS METHODS
    #########################
    

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
    


    def get_camera_state(self, rot_mat, pos):
        """
        Gets the state of the camera
        """
        # rot_adjustment = Rotation.from_quat([1, 0, 0, 0]).as_matrix()
        rot_adjustment = np.eye(3)
        # r = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
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
        # outputs["gl_z_buf_depth"] = F.interpolate(
        #     outputs["depth"].squeeze(dim=-1)[None, None, ...],
        #     size=(int(outputs["depth"].shape[0] * scale), int(outputs["depth"].shape[1] * scale)),
        #     mode="bilinear")[0, 0, :, :, None]
        
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
        
        if self.obstacle_avoidance:
            velocities = self.radial_flow.GetVelocities(depth_image=depth_image)
            return velocities
        else:
            velocities, orientations = self.vfh_planner.get_velocities_for_debugging()
            # velocities, orientations = self.vfh_planner.get_velocities_for_simple_trajectory()

        return velocities, orientations
        
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
    def save_images(self):
        
        self.image_id = 1
    
        
        c2w_list = csv_to_tensor_list("data/output136/gt_poses/poses_in_camera_frame.csv")
        os.makedirs(self.save_path, exist_ok=True)
        for c in c2w_list:
            camera_state_for_saving =  CameraState(fov=self.fov, aspect=self.aspect_ratio, c2w=c, camera_type=CameraType.PERSPECTIVE)
            outputs_for_saving = self._render_img(camera_state_for_saving)
            depth = outputs_for_saving["depth"].cpu().numpy()
            np.save(os.path.join(self.save_path,f"{self.image_id:04d}.npy"), depth)
            # current_rgb_for_saving, current_depth_for_saving = self.current_images(outputs_for_saving)

            # Saving images locally
            # cv2.imwrite(os.path.join(self.save_path, f"{self.image_id:04d}_rgb.png"), current_rgb_for_saving)
            # cv2.imwrite(os.path.join(self.save_path,f"{self.image_id:04d}_depth.png"), current_depth_for_saving)
            self.image_id+=1

        
    

    def execute(self):

        self.save_images()


def csv_to_tensor_list(csv_file_path):
    # Step 1: Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Step 2: Initialize an empty list to store the 3x4 tensors
    tensor_list = []

    # Step 3: Iterate through each row in the DataFrame and create a 3x4 tensor
    for index, row in df.iterrows():
        # Extract rotation matrix elements
        rotation_matrix = torch.tensor([
            [row['r11'], row['r12'], row['r13']],
            [row['r21'], row['r22'], row['r23']],
            [row['r31'], row['r32'], row['r33']]
        ], dtype=torch.float64)

        # Extract translation vector (x, y, z)
        translation_vector = torch.tensor([row['x'], row['y'], row['z']], dtype=torch.float64).unsqueeze(1)

        # Concatenate the rotation matrix with the translation vector to form a 3x4 matrix
        matrix_3x4 = torch.cat((rotation_matrix, translation_vector), dim=1)

        # Append the 3x4 tensor to the list
        tensor_list.append(matrix_3x4)

    # Return the list of tensors
    return tensor_list