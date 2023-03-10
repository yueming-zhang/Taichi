# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Simple matplotlib rendering of a rollout prediction against ground truth.

Usage (from parent directory):

`python -m gns.render_rollout --rollout_path={OUTPUT_PATH}/rollout_test_1.pkl`

Where {OUTPUT_PATH} is the output path passed to `train.py` in "eval_rollout"
mode.

It may require installing Tkinter with `sudo apt-get install python3.7-tk`.

"""  # pylint: disable=line-too-long

import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
from pyevtk.hl import pointsToVTK

flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file.")
flags.DEFINE_integer("step_stride", 3, help="Stride of steps to skip.")
flags.DEFINE_boolean("block_on_show", True, help="For test purposes.")

FLAGS = flags.FLAGS

TYPE_TO_COLOR = {
    3: "black",  # Boundary particles.
    0: "green",  # Rigid solids.
    7: "magenta",  # Goop.
    6: "gold",  # Sand.
    5: "blue",  # Water.
}

root = 'D:/Stanford/Taichi/data1'
# root = "//wsl.localhost/Ubuntu-20.04/home/ming/dev/gns/data/cloth"

rollout_path = f'{root}/train.npz'
def main(unused_argv):

  if rollout_path.endswith(".npz"):
    render_rollout_2d(rollout_path)
    return

  # loop through the rollout_path folder, and render_rollout for each file
  for file in os.listdir(rollout_path):
    if file.endswith(".npz"):
      render_rollout_2d(rollout_path + file)
    else:
      continue

def load_rollout(file):
  rollout_data = pickle.load(file)  
  # return rollout_data[0][1][0]
  return [i[1][0] for i in rollout_data]

def render_rollout_3d(rollout_path):
  with open(rollout_path, "rb") as file:
    rollout_data = load_rollout(file)

  fig = plt.figure(figsize=(20, 10))
  axes = fig.add_subplot(projection="3d")
  axes.set(xlim3d=(-0.5, 0.5), xlabel='X')
  axes.set(ylim3d=(-0.5, 0.5), ylabel='Y')
  axes.set(zlim3d=(-0.5, 0.5), zlabel='Z')

  plot_info = []
  path = f"{root}/vtk"       
  if not os.path.exists(path):
    os.makedirs(path)
  for i in range(len(rollout_data)):
    arr = rollout_data[i]
    if arr.shape[2] == 2:
      arr = np.concatenate([arr, np.zeros((arr.shape[0], arr.shape[1], 1))], axis=2)
    coords0 = arr[0]
    for j in range(len(arr)):
      coords = arr[j]
      disp = np.linalg.norm(coords - coords0, axis=1)
      pointsToVTK(f"{path}/points{j}", np.array(coords[:, 0]), 
                                       np.array(coords[:, 1]), 
                                       np.array(coords[:, 2]), 
                                       data = {"displacement" : disp})   
    # Append the initial positions to get the full trajectory.
    trajectory = arr
    ax = axes
    points = {
        particle_type: ax.plot([], [], [], ".", ms=2, color=color)[0]
        for particle_type, color in TYPE_TO_COLOR.items()
      }
    plot_info.append((ax, trajectory, points))

    num_steps = trajectory.shape[0]

    def update(step_i):
      outputs = []
      for _, trajectory, points in plot_info:
        for particle_type, line in points.items():
          mask = 0#rollout_data["particle_types"] == particle_type
          line.set_data(trajectory[step_i, :, 0], trajectory[step_i, :, 1])
          line.set_3d_properties(trajectory[step_i, :, 1])
          outputs.append(line)
      return outputs

    unused_animation = animation.FuncAnimation(
        fig, update,
        frames=np.arange(0, num_steps, FLAGS.step_stride), interval=10)

    gif_name = rollout_path[:-4] + ".gif"
    unused_animation.save(gif_name, dpi=80, fps=30, writer='imagemagick')
    plt.show(block=FLAGS.block_on_show)


def render_rollout_2d(rollout_path):
  with open(rollout_path, "rb") as file:
    rollout_data = load_rollout(file)

  fig, axes = plt.subplots(1, 2, figsize=(20, 10))

  plot_info = []
  path = f"{root}/vtk"       
  if not os.path.exists(path):
    os.makedirs(path)
  for i in range(len(rollout_data)):
    arr = rollout_data[i]
    if arr.shape[2] == 2:
      arr = np.concatenate([arr, np.zeros((arr.shape[0], arr.shape[1], 1))], axis=2)
    coords0 = arr[0]
    for j in range(len(arr)):
      coords = arr[j]
      disp = np.linalg.norm(coords - coords0, axis=1)
      pointsToVTK(f"{path}/points{j}", np.array(coords[:, 0]), 
                                       np.array(coords[:, 1]), 
                                       np.array(coords[:, 2]), 
                                       data = {"displacement" : disp})   
    # Append the initial positions to get the full trajectory.
    trajectory = arr
    ax = axes[0]
    ax.set_title('Ground truth')
    bounds = [[-0.5, 0.5], [-0.5, 0.5]]
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.)
    points = {
        particle_type: ax.plot([], [], ".", ms=2, color=color)[0]
        for particle_type, color in TYPE_TO_COLOR.items()
      }
    plot_info.append((ax, trajectory, points))

    num_steps = trajectory.shape[0]

    def update(step_i):
      outputs = []
      for _, trajectory, points in plot_info:
        for particle_type, line in points.items():
          mask = 0#rollout_data["particle_types"] == particle_type
          line.set_data(trajectory[step_i, :, 0], trajectory[step_i, :, 1])
          outputs.append(line)
      return outputs

    unused_animation = animation.FuncAnimation(
        fig, update,
        frames=np.arange(0, num_steps, FLAGS.step_stride), interval=10)

    gif_name = rollout_path[:-4] + ".gif"
    unused_animation.save(gif_name, dpi=80, fps=30, writer='imagemagick')
    plt.show(block=FLAGS.block_on_show)


if __name__ == "__main__":
  app.run(main)
