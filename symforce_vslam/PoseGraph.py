import numpy as np

import symforce
symforce.set_epsilon_to_symbol()
import symforce.symbolic as sf
from symforce.geo import Pose3, Rot3
from symforce.cam import camera_cal
from symforce.opt.noise_models import DiagonalNoiseModel
from symforce.opt.optimizer import Optimizer
from symforce.opt.factor import Factor
from symforce.values import Values

import cv2 as cv

import copy

class PoseGraph():

    factors: list = []
    poses: list = [sf.Pose3.identity()]
    odoms: list = []
    tag_ids = [0, 1, 2, 3, 4, 5]
    n_detections = []
    detections: list = []
    initial_tag_poses: list = []
    
    camera_K = np.array([
        [491.08, 0, 372.68],
        [0, 490.28, 221.30],
        [0, 0, 1]
    ])
    camera_cal: camera_cal
    g_robot_cam: sf.Pose3

    def __init__(self):
        # Create hard-coded camera calibration object
        focal_length = [self.camera_K[0, 0], self.camera_K[1, 1]]
        principal_point = [self.camera_K[0, 2], self.camera_K[1, 2]]
        self.camera_cal = sf.LinearCameraCal(focal_length=focal_length, principal_point = principal_point)

        # Create hard-coded transformation from robot center to camera
        R_robot_cam = sf.Rot3.from_yaw_pitch_roll(-np.pi/2, 0, -np.pi/2)
        t_robot_cam = sf.Vector3([-0.1016, 0, 0.0889])
        self.g_robot_cam = sf.Pose3(R_robot_cam, t_robot_cam)

        self.n_detections = [0 for id in self.tag_ids]
        self.detections = [[] for id in self.tag_ids]
        self.initial_tag_poses = [sf.Pose3.identity()] * len(self.tag_ids)

    def get_tag_corner_residual(self, i_corner: np.int_):
        ## Get the position of the tags in the camera frame
        # Define the displacements from the tag center to each tag corner
        R = sf.Rot3.identity()
        w = 0.203
        # Corner order: start in bottom left (-x, -y) and go ccw
        g_tag_c1 = sf.Pose3(R, sf.Vector3([-w/2, w/2, 0]))
        g_tag_c2 = sf.Pose3(R, sf.Vector3([w/2, w/2, 0]))
        g_tag_c3 = sf.Pose3(R, sf.Vector3([w/2, -w/2, 0]))
        g_tag_c4 = sf.Pose3(R, sf.Vector3([-w/2, -w/2, 0]))
        g_tag_corners = [g_tag_c1, g_tag_c2, g_tag_c3, g_tag_c4]
        g_tag_ci = g_tag_corners[i_corner]

        camera_noise = DiagonalNoiseModel.from_sigmas(sf.Vector2([20, 20]))
        def residual_tag_obs(robot_pose: sf.Pose3, tag_pose: sf.Pose3, corners_px_measured: sf.M12, epsilon: sf.Scalar):
            # Measure the error between the expected pixel coordinates of the tag corners
            # versus the actual measured ones.
            camera_pose = robot_pose * self.g_robot_cam

            # Compute the expected tag corner positions in the camera frame
            # based on the hypothesized camera and tag poses.
            g_cam_tag = camera_pose.inverse() * tag_pose
            g_cam_corner = g_cam_tag * g_tag_ci
        
            # Project the tag corner positions into the camera frame using the camera model
            corners_px_expected = self.camera_cal.pixel_from_camera_point(g_cam_corner.t, epsilon)[0]
        
            return camera_noise.whiten(sf.Vector2(corners_px_measured.T - corners_px_expected))
        return residual_tag_obs

    def add_prior_factor(self, prior_pose: sf.Pose3):
        prior_noise = DiagonalNoiseModel.from_sigmas(sf.Vector6([
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02
        ]))
        def prior_residual(measured_pose: sf.Pose3, epsilon: sf.Scalar):
            # Prior residual = prior_pose - measured_pose
            pose_diff = measured_pose.local_coordinates(prior_pose, epsilon)
            return prior_noise.whiten(sf.Vector6(pose_diff))

        self.factors.append(Factor(
            residual=prior_residual,
            keys=["poses[0]", "epsilon"]
        ))

    def add_tag_factors(self, msg_detections):
        for detection in msg_detections:
            tag_id = detection.id
            i_pose = len(self.poses) - 1
            
            corners_i = []
            for i, corner in enumerate(detection.corners):
                self.factors.append(Factor(
                    residual=self.get_tag_corner_residual(i),
                    keys=[
                        f"poses[{i_pose}]",
                        f"tag_poses[{tag_id}]",
                        f"detections[{tag_id}][{self.n_detections[tag_id]}][{i}]",
                        "epsilon"
                    ]
                ))

                corners_i.append(sf.M12([corner.x, corner.y]))
                
            if not self.detections[tag_id]:
                # Initia sighting of the tag: initializing tag pose based on odometry and pnp
                w = 0.203
                objps = np.array([
                    [-w/2, w/2, 0],
                    [w/2, w/2, 0],
                    [w/2, -w/2, 0],
                    [-w/2, -w/2, 0],
                ])
                imageps = np.array(corners_i, dtype=np.float64)
                
                pnpsoln = cv.solvePnP(objps, imageps, self.camera_K, np.array([]), flags=cv.SOLVEPNP_SQPNP)
                theta = np.linalg.norm(pnpsoln[1])
                g_tag_R_i = sf.Rot3.from_angle_axis(theta, sf.Vector3(pnpsoln[1]) / theta)
                g_tag_t_i = sf.Vector3(pnpsoln[2])
                g_cam_to_tag_i = sf.Pose3(g_tag_R_i, g_tag_t_i)
                g_tag_i = self.poses[-1] * self.g_robot_cam * g_cam_to_tag_i
                
                self.initial_tag_poses[tag_id] = g_tag_i
                
            self.detections[tag_id].append(corners_i)
            # Count a new detection for this tag
            self.n_detections[tag_id] += 1

    def add_odometry_factor(self, delta_pose):
        odometry_noise = DiagonalNoiseModel.from_sigmas(sf.Vector6([
            0.05, 0.05, 0.05, np.deg2rad(0.001), np.deg2rad(0.001), np.deg2rad(0.5)
        ]))

        # Residual function generator
        def residual_between(pose_1: sf.Pose3, pose_2: sf.Pose3, rdelta_measured: sf.Pose3, epsilon: sf.Scalar):
            # Compute the expected body-frame displacement
            # pose_1 * rdelta = pose_2
            rdelta_expected = pose_1.inverse() * pose_2

            # The "error": rdelta_expected ⊟ rdelta_measured
            error = rdelta_measured.local_coordinates(rdelta_expected, epsilon)
            return odometry_noise.whiten(sf.Vector6(error))

        # Code to add the factor
        self.odoms.append(delta_pose)
        self.poses.append(self.poses[-1] * delta_pose)

        i_pose = len(self.poses) - 2    # Index of the first pose: -1 for converting length to zero-based index, and -1 for fetching the first one
        self.factors.append(Factor(
            residual=residual_between,
            keys=[f"poses[{i_pose}]", f"poses[{i_pose+1}]", f"odoms[{i_pose}]", "epsilon"]
        ))

    def make_initial_values(self):
        # Populate the detection arrays with a 0 for any tag without any detections
        # This is because symforce expects a certain full-dimensionality for all parts of an array
        initial_values = Values(
            poses = self.poses,
            odoms = self.odoms,
            detections = self.detections,
            tag_poses = self.initial_tag_poses,
            epsilon = sf.numeric_epsilon
        )

        modified_vals = Values.copy(initial_values)

        for i, detection_list in enumerate(self.detections):
            if self.n_detections[i] == 0:
                modified_vals["detections"][i] = [[0]]
                modified_vals["tag_poses"][i] = sf.Pose3.identity()

        return modified_vals

    def solve(self, initial_values=None):
        if initial_values is None:
            initial_values = self.make_initial_values()

        ## Solve the optimization problem
        n_poses = len(self.poses)
        pose_keys = [f"poses[{i}]" for i in range(n_poses)]
        tag_keys = [f"tag_poses[{tag_id}]" for tag_id in self.tag_ids if self.n_detections[tag_id] != 0]

        optimizer = Optimizer(
            factors=self.factors,
            optimized_keys=pose_keys+tag_keys,
            # So that we save more information about each iteration, to visualize later:
            debug_stats=True,
        )
        result = optimizer.optimize(initial_values)
        return result