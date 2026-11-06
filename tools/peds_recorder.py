#!/usr/bin/env python3
"""Record the simulator's people (arena_people_msgs/Pedestrians) for episode videos. CONTAINER only.

  python3 tools/peds_recorder.py --out <run_dir>              # first topic ending in arena_peds
  python3 tools/peds_recorder.py --out <run_dir> --topic /arena_peds --every 3

Writes <out>/peds.csv: stamp_s (simulation time from the header), wall_s, id, name, x, y, yaw, speed.
Planner-neutral: VLN planners never see people positions, so render_vln.py takes them from here.
Uses the ROS Python of the container (rclpy), not the planner venv. Ctrl-c to stop.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import signal
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class PedsRecorder(Node):
    def __init__(self, out: pathlib.Path, topic: str, every: int):
        super().__init__("bench3_peds_recorder")
        from arena_people_msgs.msg import Pedestrians  # noqa: PLC0415

        self._msg = Pedestrians
        out.mkdir(parents=True, exist_ok=True)
        self.fh = (out / "peds.csv").open("a", newline="")
        self.w = csv.writer(self.fh)
        if self.fh.tell() == 0:
            self.w.writerow(["stamp_s", "wall_s", "id", "name", "x", "y", "yaw", "speed"])
        self.topic, self.every, self.n, self.rows = topic, max(1, every), 0, 0
        self.sub = None
        self.timer = self.create_timer(0.5, self._connect)

    def _connect(self) -> None:
        if self.sub is not None:
            return
        topic = self.topic
        if not topic:
            cands = [n for n, t in self.get_topic_names_and_types() if n.endswith("arena_peds") and "arena_people_msgs/msg/Pedestrians" in t]
            if not cands:
                return
            topic = cands[0]
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(self._msg, topic, self._on_msg, qos)
        self.get_logger().info(f"recording {topic} -> {self.fh.name}")

    def _on_msg(self, msg) -> None:
        self.n += 1
        if self.n % self.every:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        wall = time.time()
        for p in msg.pedestrians:
            q = p.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            v = math.hypot(p.twist.linear.x, p.twist.linear.y)
            self.w.writerow([f"{stamp:.3f}", f"{wall:.3f}", p.id, p.name, f"{p.pose.position.x:.3f}", f"{p.pose.position.y:.3f}", f"{yaw:.3f}", f"{v:.3f}"])
            self.rows += 1
        if self.n % (self.every * 30) == 0:
            self.fh.flush()

    def close(self) -> None:
        self.fh.flush()
        self.fh.close()
        self.get_logger().info(f"{self.rows} rows written")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run dir; peds.csv is appended there")
    ap.add_argument("--topic", default="", help="default: first topic ending in arena_peds")
    ap.add_argument("--every", type=int, default=1, help="keep every n-th message")
    a = ap.parse_args()
    rclpy.init()
    node = PedsRecorder(pathlib.Path(a.out), a.topic, a.every)
    signal.signal(signal.SIGTERM, lambda *_: rclpy.shutdown())
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
