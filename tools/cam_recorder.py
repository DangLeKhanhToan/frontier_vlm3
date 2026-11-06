#!/usr/bin/env python3
"""Record the 30 Hz recording camera as JPEG frames for episode videos. CONTAINER only.

  python3 tools/cam_recorder.py --out /opt/arena_ws/data/vlm_video/<run_label>
  python3 tools/cam_recorder.py --out ... --topic /jackal/vln_camera/rec_image

Writes <out>/frames/000001.jpg ... and <out>/index.csv with one line per frame:
  frame, stamp_s (simulation time from the image header), wall_s (host clock on receipt)
render_episode.py uses the (wall_s, stamp_s) pairs to put the planner logs (host clock)
on the simulation-time axis. Start it before `arena launch`: it waits for the topic.
Raw 640x480 would be ~27 MB/s in a rosbag; JPEG q85 is ~1 MB/s.

Needs the robot with the recording camera: robot:='jackal[camera=vln_rgbd_rec]'.
Uses the ROS Python of the container (rclpy, numpy, cv2), not the planner venv.
"""

from __future__ import annotations

import argparse
import pathlib
import signal
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class Recorder(Node):
    def __init__(self, out: pathlib.Path, topic: str, suffix: str, quality: int, every: int):
        super().__init__("fvlm3_cam_recorder")
        self.out = out
        (out / "frames").mkdir(parents=True, exist_ok=True)
        self.index = open(out / "index.csv", "a", buffering=1)
        if self.index.tell() == 0:
            self.index.write("frame,stamp_s,wall_s\n")
        self.n = len(list((out / "frames").glob("*.jpg")))
        self.quality = quality
        self.every = max(1, every)
        self.seen = 0
        self.topic = topic
        self.suffix = suffix
        self.sub = None
        self.last_report = time.time()
        self.timer = self.create_timer(1.0, self._discover)

    def _discover(self) -> None:
        if self.sub is None:
            topic = self.topic
            if not topic:
                cands = [n for n, types in self.get_topic_names_and_types() if n.endswith(self.suffix) and "sensor_msgs/msg/Image" in types]
                if not cands:
                    if time.time() - self.last_report > 10:
                        self.get_logger().info(f"waiting for a topic ending in {self.suffix!r} ...")
                        self.last_report = time.time()
                    return
                if len(cands) > 1:
                    self.get_logger().warn(f"several recording cameras {cands}; using {cands[0]} (pass --topic)")
                topic = cands[0]
            # match the publisher: a reliable subscriber gets nothing from a best-effort publisher,
            # and best effort drops many 900 kB frames (lost UDP fragments)
            pubs = self.get_publishers_info_by_topic(topic)
            best_effort = any(i.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT for i in pubs)
            rel = ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE
            qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST, reliability=rel)
            self.sub = self.create_subscription(Image, topic, self._on_image, qos)
            self.get_logger().info(f"recording {topic} ({rel.name.lower()}) -> {self.out}")
        elif time.time() - self.last_report > 30:
            self.get_logger().info(f"{self.n} frames")
            self.last_report = time.time()

    def _on_image(self, msg: Image) -> None:
        wall = time.time()
        self.seen += 1
        if (self.seen - 1) % self.every:
            return
        enc = msg.encoding.lower()
        ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(enc)
        if ch is None:
            self.get_logger().error(f"unsupported encoding {msg.encoding}", throttle_duration_sec=10.0)
            return
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
        if enc in ("rgb8", "rgba8"):
            a = cv2.cvtColor(a, cv2.COLOR_RGBA2BGR if ch == 4 else cv2.COLOR_RGB2BGR)
        elif enc == "bgra8":
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
        self.n += 1
        cv2.imwrite(str(self.out / "frames" / f"{self.n:06d}.jpg"), a, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.index.write(f"{self.n},{stamp:.4f},{wall:.4f}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topic", default="", help="image topic; default: first topic ending in --suffix")
    ap.add_argument("--suffix", default="/rec_image")
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--every", type=int, default=1, help="keep every n-th frame")
    a = ap.parse_args()
    rclpy.init()
    node = Recorder(pathlib.Path(a.out), a.topic, a.suffix, a.quality, a.every)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.get_logger().info(f"stopped, {node.n} frames in {node.out}")
        node.index.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
