#!/bin/usr/env python3
import rosbag2_py
from rclpy.serialization import deserialize_message
import importlib

import numpy as np

import symforce
symforce.set_epsilon_to_symbol()
import symforce.symbolic as sf

from PoseGraph import PoseGraph

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

# Supporting functions
def pose3_msg_to_sf(msg):
    q_xyz = [msg.orientation.x, msg.orientation.y, msg.orientation.z]
    w = msg.orientation.w
    quat = sf.Quaternion(xyz=sf.Vector3(q_xyz), w=w)
    R = sf.Rot3(quat)

    posn = [msg.position.x, msg.position.y, msg.position.z]
    t = sf.Vector3(posn)
    return sf.Pose3(R, t)

def main():
    ## Pose graph length parameters
    start_factor = 300
    n_factors = 7500

    # Fetch teh bag file 
    path = "../bags/dataset_drive_square_no_vid"
    msgs, msg_type_table = get_msgs_from_bag(path)

    f_deserialize_msg = lambda msg: deserialize_message(msg[1], msg_type_table[msg[0]])

    # Build the pose graph by looping through all messages
    pg = PoseGraph()
    last_odom_pose = sf.Pose3.identity()
    last_odom_time = 0
    i_pose = 0
    for i in range(start_factor, start_factor + n_factors):
        topic, bdata, time = msgs[i]
        data = f_deserialize_msg(msgs[i])

        if topic == "/odom":
            ## Record the sensor data into the list of displacements
            pose = pose3_msg_to_sf(data.pose.pose)
            if i_pose == 0:
                delta = sf.Pose3.identity()
            else:
                # Otherwise, record the displacement from the last pose
                # last_pose * delta = pose
                # => delta = inv(last_pose) * pose
                delta = last_odom_pose.inverse() * pose
            pg.add_odometry_factor(delta)
            
            last_odom_pose = pose
            last_odom_time = time

            i_pose += 1
            
        elif topic == "/detections" and (time - last_odom_time) * 1e-9 < 0.02:
            pg.add_tag_factors(data.detections)

    ## Solve the optimization problem
    result, initial_values = pg.solve()
    plot_optimization_results(initial_values, result)


if __name__ == "__main__":
    main()