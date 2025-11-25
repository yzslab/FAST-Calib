import os
import time
import argparse
import subprocess
import cv2
import rosbag
from cv_bridge import CvBridge
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--bag", type=str, default=None)
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--duration", "-d", type=int, default=30)
parser.add_argument("--name", "-n", type=str, default=None)
parser.add_argument("--image_topic", "-t", type=str, default="/left_camera/image/compressed")
parser.add_argument("--fisheye", action="store_true", default=False)
args = parser.parse_args()

IMAGE_TOPIC = args.image_topic

if args.bag is None:
    name = args.name
    if name is None:
        name = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "calibrations", name))
    bag_file_path = os.path.join(output_dir, "data.bag")
else:
    output_dir = os.path.realpath("{}-calibration".format(args.bag))
    if args.name is not None:
        output_dir += "-{}".format(args.name)
    bag_file_path = args.bag
os.makedirs(output_dir, exist_ok=True)

if args.image is None:
    image_file_path = os.path.join(output_dir, "image.jpg")
else:
    image_file_path = args.image
config_file_path = os.path.join(output_dir, "config.yaml")
calibration_output_path = os.path.join(output_dir, "result")
os.makedirs(calibration_output_path, exist_ok=True)

if args.bag is None:
    assert os.path.exists(bag_file_path) is False
    print("Recording to {}...".format(bag_file_path))
    subprocess.call([
        "rosbag",
        "record",
        "--duration={}s".format(args.duration),
        "--output-name={}".format(bag_file_path),
        "/livox/lidar",
        "/livox/imu",
        IMAGE_TOPIC,
    ])

if args.image is None:
    print("Extracting an image...")
    bag = rosbag.Bag(bag_file_path, "r")
    assert bag.get_message_count(IMAGE_TOPIC) > 0
    _, first_image_message, _ = next(bag.read_messages(topics=[IMAGE_TOPIC]))

    msg = first_image_message
    assert "jpeg" in msg.format
    stamp = msg.header.stamp.to_nsec()
    with open(image_file_path, "wb") as f:
        f.write(msg.data)
        previous_stamp = stamp

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

if args.fisheye or config.get("fisheye", False):
    print("Fisheye mode")

    distorted_image = cv2.imread(image_file_path)

    import numpy as np
    import camera_calibration_utils
    undistorter = camera_calibration_utils.FisheyeCameraUndistorter(
        DIM=np.asarray([distorted_image.shape[1], distorted_image.shape[0]]),
        K=np.asarray([
            [config["fx"], 0., config["cx"]],
            [0., config["fy"], config["cy"]],
            [0., 0., 1.],
        ]),
        D=np.asarray([config["k1"], config["k2"], config["p1"], config["p2"]]),
        balance=0,
    )
    undistorted_image = undistorter.undistort(distorted_image)
    undistorted_image_path = os.path.join(os.path.dirname(image_file_path), "image-undistorted.png")
    cv2.imwrite(
        undistorted_image_path,
        undistorted_image,
    )

    image_file_path = undistorted_image_path
    config["k1"] = 0.
    config["k2"] = 0.
    config["p1"] = 0.
    config["p2"] = 0.


config["bag_path"] = bag_file_path
config["image_path"] = image_file_path
config["output_path"] = calibration_output_path
with open(config_file_path, "w") as f:
    yaml.dump(config, f)

print("Calibrating...")
os.execvp("roslaunch", [
    "roslaunch",
    "fast_calib",
    "auto.launch",
    "param_file:={}".format(config_file_path),
])
