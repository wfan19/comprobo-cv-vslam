#!/bin/usr/env python3
import rosbag2_py
from rclpy.serialization import deserialize_message
import importlib

import numpy as np

import symforce
symforce.set_epsilon_to_symbol()
import symforce.symbolic as sf
from symforce.cam import camera_cal
from symforce.opt.noise_models import DiagonalNoiseModel
from symforce.opt.optimizer import Optimizer
from symforce.opt.factor import Factor
from symforce.values import Values

import cv2 as cv

import plotly.graph_objects as go

def get_type_from_str(type_str: str):
    # dynamic load message package
    pkg = importlib.import_module(".".join(type_str.split("/")[:-1]))
    return eval(f"pkg.{type_str.split('/')[-1]}")    

def get_msgs_from_bag(path):
    bag_path = path
    # Copied the following from https://qiita.com/nonanonno/items/8f7bce03953709fd5af9
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")

    # Create a reader object for the bag
    sr = rosbag2_py.SequentialReader()
    sr.open(storage_options, converter_options)
    sr.set_filter(rosbag2_py.StorageFilter(topics=["/odom", "/detections", "/camera/camera_info"])) 

    # Create a dictionary for the message type name string and the actual class
    msg_type_table = {}
    for topic in sr.get_all_topics_and_types():
        msg_type_table[topic.name] = get_type_from_str(topic.type)

    # Go through the bag pulling out messages
    sr.seek(0)
    msgs = []
    while sr.has_next():
        msgs.append(sr.read_next())

    # Store all the messages besides the camera_infos (there's too many of them)
    msgs_no_cam = [msg for msg in msgs if not (msg[0] == "/camera/camera_info")] 

    # Pull out the first camera info message
    cam_info_msg = next(
        deserialize_message(msg[1], msg_type_table[msg[0]])
        for msg in msgs if msg[0] == "/camera/camera_info"
    )

    # Insert it into the front
    msgs_no_cam.insert(0, cam_info_msg)

    return msgs_no_cam, msg_type_table

def plot_optimization_results(initial_values, result):
    initial_poses = initial_values["poses"]
    robot_poses_initial = initial_poses
    robot_t_initial = np.array([pose.t for pose in robot_poses_initial], dtype=np.float32)

    robot_poses_optimized = result.optimized_values["poses"]
    robot_t_optimized = np.array([pose.t for pose in robot_poses_optimized], dtype=np.float32)

    tag_poses = result.optimized_values["tag_poses"]
    tag_t = np.array([pose.t for pose in tag_poses])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x = robot_t_initial[:, 0],
        y = robot_t_initial[:, 1],
        z = robot_t_initial[:, 2],
        name="initial"
    ))

    fig.add_trace(go.Scatter3d(
        x = robot_t_optimized[:, 0],
        y = robot_t_optimized[:, 1],
        z = robot_t_optimized[:, 2],
        name="optimized"
    ))

    fig.add_trace(go.Scatter3d(
        x = tag_t[:, 0],
        y = tag_t[:, 1],
        z = tag_t[:, 2],
        mode="markers",
        name="tags",
        marker={
            "color": "black"
        }
    ))

    fig.update_layout(scene_aspectmode="data", scene=dict(zaxis=dict(range=[-0.25, 0.25])))
    fig.show()

def main():
    path = "../bags/dataset_drive_square_no_vid"
    msgs, msg_type_table = get_msgs_from_bag(path)

    f_deserialize_msg = lambda msg: deserialize_message(msg[1], msg_type_table[msg[0]])

    cam_info_msg = msgs[0]
    camera_K = np.reshape(cam_info_msg.k, (3, 3))
    focal_length = [camera_K[0, 0], camera_K[1, 1]]
    principal_point = [camera_K[0, 2], camera_K[1, 2]]
    camera_cal = sf.LinearCameraCal(focal_length=focal_length, principal_point = principal_point)

    ## Define the residual functions
    odometry_noise = DiagonalNoiseModel.from_sigmas(sf.Vector6([
        0.01, 0.01, 0.01, np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)
    ]))
    # odometry_noise = DiagonalNoiseModel.from_sigmas(sf.Vector6([
    #     1, 1, 1, 1, 1, 1,
    # ]))
    def residual_between(pose_1: sf.Pose3, pose_2: sf.Pose3, rdelta_measured: sf.Pose3, epsilon: sf.Scalar):
        # Compute the expected body-frame displacement
        # pose_1 * rdelta = pose_2
        rdelta_expected = pose_1.inverse() * pose_2

        # The "error": rdelta_expected ⊟ rdelta_measured
        error = rdelta_measured.local_coordinates(rdelta_expected, epsilon)
        return odometry_noise.whiten(sf.Vector6(error))

    R_robot_cam = sf.Rot3.from_yaw_pitch_roll(-np.pi/2, 0, -np.pi/2)
    t_robot_cam = sf.Vector3([-0.1016, 0, 0.0889])
    g_robot_cam = sf.Pose3(R_robot_cam, t_robot_cam)
    camera_noise = DiagonalNoiseModel.from_sigmas(sf.Vector2([10, 10]))
    # camera_noise = DiagonalNoiseModel.from_sigmas(sf.Vector2([1, 1]))
    # TODO Rewrite this to be one residual vector / factor per tag, not four
    def get_tag_corner_residual(i_corner: np.int_):
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

        def residual_tag_obs(robot_pose: sf.Pose3, tag_pose: sf.Pose3, corners_px_measured: sf.M12, epsilon: sf.Scalar):
            # Measure the error between the expected pixel coordinates of the tag corners
            # versus the actual measured ones.
            camera_pose = robot_pose * g_robot_cam

            # Compute the expected tag corner positions in the camera frame
            # based on the hypothesized camera and tag poses.
            g_cam_tag = camera_pose.inverse() * tag_pose
            g_cam_corner = g_cam_tag * g_tag_ci
        
            # Project the tag corner positions into the camera frame using the camera model
            corners_px_expected = camera_cal.pixel_from_camera_point(g_cam_corner.t, epsilon)[0]
        
            return camera_noise.whiten(sf.Vector2(corners_px_measured.T - corners_px_expected))
        return residual_tag_obs
    
    ## Build the pose graph and initialize initial values
    start_factor = 300
    n_factors = 7500
    tag_ids = [0, 1, 2, 3, 4, 5]

    # Supporting functions
    def pose3_msg_to_sf(msg):
        q_xyz = [msg.orientation.x, msg.orientation.y, msg.orientation.z]
        w = msg.orientation.w
        quat = sf.Quaternion(xyz=sf.Vector3(q_xyz), w=w)
        R = sf.Rot3(quat)

        posn = [msg.position.x, msg.position.y, msg.position.z]
        t = sf.Vector3(posn)
        return sf.Pose3(R, t)

    ## Initialize storage vars
    factors = []
    # Store odometry sensor data
    i_pose = 0
    odoms = []
    poses = [sf.Pose3.identity()]
    last_pose = sf.Pose3.identity()
    # Store tag detections
    n_detections = [0 for id in tag_ids]
    detections = [[] for id in tag_ids]
    last_odom_time = 0
    initial_tag_poses = [sf.Pose3.identity()] * len(tag_ids)

    for i in range(start_factor, start_factor + n_factors):
        topic, bdata, time = msgs[i]
        data = f_deserialize_msg(msgs[i])

        if topic == "/odom":
            ## Add a between-factor to the factor graph
            factors.append(Factor(
                residual=residual_between,
                keys=[f"poses[{i_pose}]", f"poses[{i_pose+1}]", f"odoms[{i_pose}]", "epsilon"]
            ))

            ## Record the sensor data into the list of displacements
            pose = pose3_msg_to_sf(data.pose.pose)
            if i_pose == 0:
                # If first displacement measurement, simply record 0
                odoms.append(sf.Pose3.identity())
                poses.append(sf.Pose3.identity())
            else:
                # Otherwise, record the displacement from the last pose
                # last_pose * delta = pose
                # => delta = inv(last_pose) * pose
                delta = last_pose.inverse() * pose
                odoms.append(delta)
                poses.append(poses[-1] * delta)
            
            last_pose = pose
            i_pose = i_pose + 1
            last_odom_time = time
            
        elif topic == "/detections" and (time - last_odom_time) * 1e-9 < 0.02:
            for detection in data.detections:
                tag_id = detection.id
                
                corners_i = []
                for i, corner in enumerate(detection.corners):
                    factors.append(Factor(
                        residual=get_tag_corner_residual(i),
                        keys=[
                            f"poses[{i_pose}]",
                            f"tag_poses[{tag_id}]",
                            f"detections[{tag_id}][{n_detections[tag_id]}][{i}]",
                            "epsilon"
                        ]
                    ))

                    corners_i.append(sf.M12([corner.x, corner.y]))
                    
                if not detections[tag_id]:
                    # Initia sighting of the tag: initializing tag pose based on odometry and pnp
                    w = 0.203
                    objps = np.array([
                        [-w/2, w/2, 0],
                        [w/2, w/2, 0],
                        [w/2, -w/2, 0],
                        [-w/2, -w/2, 0],
                    ])
                    imageps = np.array(corners_i, dtype=np.float64)
                    
                    pnpsoln = cv.solvePnP(objps, imageps, camera_K, np.array([]), flags=cv.SOLVEPNP_SQPNP)
                    theta = np.linalg.norm(pnpsoln[1])
                    g_tag_R_i = sf.Rot3.from_angle_axis(theta, sf.Vector3(pnpsoln[1]) / theta)
                    g_tag_t_i = sf.Vector3(pnpsoln[2])
                    g_cam_to_tag_i = sf.Pose3(g_tag_R_i, g_tag_t_i)
                    g_tag_i = poses[-1] * g_robot_cam * g_cam_to_tag_i
                    
                    initial_tag_poses[tag_id] = g_tag_i
                    
                detections[tag_id].append(corners_i)
                # Count a new detection for this tag
                n_detections[tag_id] += 1

    n_poses = i_pose + 1
    initial_poses = poses

    for i, detection_list in enumerate(detections):
        if n_detections[i] == 0:
            detections[i] = [[0]]
            initial_tag_poses[i] = sf.Pose3.identity()

    initial_values = Values(
        poses = initial_poses,
        odoms = odoms,
        detections = detections,
        tag_poses = initial_tag_poses,
        epsilon = sf.numeric_epsilon
    )

    ## Solve the optimization problem
    pose_keys = [f"poses[{i}]" for i in range(n_poses)]
    tag_keys = [f"tag_poses[{tag_id}]" for tag_id in tag_ids if n_detections[tag_id] != 0]

    optimizer = Optimizer(
        factors=factors,
        optimized_keys=pose_keys+tag_keys,
        # So that we save more information about each iteration, to visualize later:
        debug_stats=True,
    )
    result = optimizer.optimize(initial_values)

    plot_optimization_results(initial_values, result)


if __name__ == "__main__":
    main()