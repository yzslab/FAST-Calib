import os
import time
import argparse
import subprocess
import cv2
import rosbag
from cv_bridge import CvBridge
import yaml

IMAGE_TOPIC = "/left_camera/image/compressed"


parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--duration", "-d", type=int, default=30)
args = parser.parse_args()

output_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "calibrations", time.strftime('%Y%m%d_%H%M%S')))
os.makedirs(output_dir, exist_ok=True)

bag_file_path = os.path.join(output_dir, "data.bag")
image_file_path = os.path.join(output_dir, "image.png")
config_file_path = os.path.join(output_dir, "config.yaml")
calibration_output_path = os.path.join(output_dir, "result")
os.makedirs(calibration_output_path, exist_ok=True)

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

print("Extracting an image...")
bag = rosbag.Bag(bag_file_path, "r")
assert bag.get_message_count(IMAGE_TOPIC) > 0
_, first_image_message, _ = next(bag.read_messages(topics=[IMAGE_TOPIC]))
bridge = CvBridge()
image = bridge.compressed_imgmsg_to_cv2(first_image_message, desired_encoding="passthrough")
cv2.imwrite(image_file_path, image)

with open(args.config, "r") as f:
    config = yaml.safe_load(f)
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
