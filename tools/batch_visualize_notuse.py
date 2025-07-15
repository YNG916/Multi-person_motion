#!/usr/bin/env python3
"""

用法示例：
  python batch_visualize.py \
    --npy_dir1 /cvhci/temp/yyang/InterGen/data/motions_change_speed/person1 \
    --npy_dir2 /cvhci/temp/yyang/InterGen/data/motions_change_speed/person2 \
    --out_dir visualize_results \
    --fps 30 \
    --radius 4
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

# 22 关节的骨骼拓扑 (t2m)
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20]
]

def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
    verts = [
        [minx, miny, minz],
        [minx, miny, maxz],
        [maxx, miny, maxz],
        [maxx, miny, minz]
    ]
    plane = Poly3DCollection([verts])
    plane.set_facecolor((0.5,0.5,0.5,0.5))
    ax.add_collection3d(plane)

def plot_3d_motion(save_path, kinematic_tree, mp_joints, title,
                   figsize=(10,10), fps=30, radius=4):
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    # 标题换行
    words = title.split(' ')
    if len(words)>20:
        title = '\n'.join([' '.join(words[:10]), ' '.join(words[10:20]), ' '.join(words[20:])])
    elif len(words)>10:
        title = '\n'.join([' '.join(words[:10]), ' '.join(words[10:])])
    fig.suptitle(title, fontsize=16)

    # 取最短帧数
    frame_number = min(j.shape[0] for j in mp_joints)

    # 颜色
    base_colors = ['red','green','black','red','blue'] + ['darkblue']*5 + ['darkred']*5
    mp_colors = [[base_colors[i]]*len(kinematic_tree) for i in range(len(mp_joints))]

    # 贴地
    mp_data = []
    for joints in mp_joints:
        data = joints.copy()
        data[:,:,1] -= data[:,:,1].min()
        mp_data.append(data)

    def update(f):
        ax.clear()
        ax.view_init(elev=120, azim=-90); ax.dist=6
        plot_xzPlane(ax, -radius, radius, 0, -radius, radius)
        for pid, data in enumerate(mp_data):
            for ci, chain in enumerate(kinematic_tree):
                pts = data[f, chain, :]
                ax.plot(pts[:,0], pts[:,1], pts[:,2],
                        lw=2 if ci<2 else 1, color=mp_colors[pid][ci])
        ax.set_xlim(-radius, radius)
        ax.set_ylim(0, radius*0.7)
        ax.set_zlim(-radius, radius*0.7)
        ax.set_axis_off()

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps)
    ani.save(save_path, fps=fps)
    plt.close(fig)

def visualize_pair(npy_dir1, npy_dir2, out_dir, fps, radius):
    os.makedirs(out_dir, exist_ok=True)
    files1 = sorted(f for f in os.listdir(npy_dir1) if f.endswith('.npy'))
    files2 = sorted(f for f in os.listdir(npy_dir2) if f.endswith('.npy'))
    assert files1==files2, "person1 与 person2 文件不匹配"

    for fname in files1:
        path1 = os.path.join(npy_dir1, fname)
        path2 = os.path.join(npy_dir2, fname)
        d1 = np.load(path1)  # shape (T, D)
        d2 = np.load(path2)
        T1, D1 = d1.shape; T2, D2 = d2.shape
        assert T1==T2, f"{fname} 帧数不一致"
        # 只取前 22 关节位置
        J = 22
        p1 = d1[:, :J*3].reshape(T1, J, 3)
        p2 = d2[:, :J*3].reshape(T2, J, 3)

        out_mp4 = os.path.join(out_dir, fname.replace('.npy','_pair.mp4'))
        print(f"[渲染] {fname} → {os.path.basename(out_mp4)}")
        plot_3d_motion(
            save_path=out_mp4,
            kinematic_tree=t2m_kinematic_chain,
            mp_joints=[p1, p2],
            title=fname.replace('.npy',''),
            fps=fps,
            radius=radius
        )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_dir1', required=True, help="person1 npy 目录")
    parser.add_argument('--npy_dir2', required=True, help="person2 npy 目录")
    parser.add_argument('--out_dir',  required=True, help="输出 mp4 目录")
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--radius', type=float, default=4)
    args = parser.parse_args()

    visualize_pair(
        npy_dir1=args.npy_dir1,
        npy_dir2=args.npy_dir2,
        out_dir=args.out_dir,
        fps=args.fps,
        radius=args.radius
    )