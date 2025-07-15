"""
python batch_visualize_new.py \
  --npy_dir1 /cvhci/temp/yyang/InterGen/data/motions_change_speed/person1 \
  --npy_dir2 /cvhci/temp/yyang/InterGen/data/motions_change_speed/person2 \
  --out_dir results/ \
  --fps 30 \
  --radius 4

"""

import os
import sys
import argparse
import numpy as np

# 如果 plot_3d_motion 定义在 tools/plot_script.py，就把上一级加入 sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.plot_script import plot_3d_motion  # 引入你给的函数
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 只为安全，plot_3d_motion 已封装好

# 对应 t2m 的 22 关节骨骼拓扑
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20]
]

def visualize_pair(npy_dir1, npy_dir2, out_dir, fps, radius):
    os.makedirs(out_dir, exist_ok=True)
    files1 = sorted(f for f in os.listdir(npy_dir1) if f.endswith(".npy"))
    files2 = sorted(f for f in os.listdir(npy_dir2) if f.endswith(".npy"))
    assert files1 == files2, "person1 与 person2 文件列表不一致！"

    for fname in files1:
        # p1 = os.path.join(npy_dir1, fname)
        # p2 = os.path.join(npy_dir2, fname)
        # d1 = np.load(p1)
        # d2 = np.load(p2)
        # T1, D1 = d1.shape
        # T2, D2 = d2.shape
        # assert T1 == T2, f"{fname} 两人帧数不同！"

        # # 只取前 22 关节 (x,y,z)
        # J = 22
        # j1 = d1[:, :J*3].reshape(T1, J, 3)
        # j2 = d2[:, :J*3].reshape(T2, J, 3)
        
        path1 = os.path.join(npy_dir1, fname)
        path2 = os.path.join(npy_dir2, fname)
        raw1 = np.load(path1)  # shape (T, D_raw)
        raw2 = np.load(path2)
        assert raw1.shape[0] == raw2.shape[0], f"{fname} 两人帧数不同！"

        # 模仿 plot_t2m 的切片与 reshape，只取前 22*3 列
        mp_joint = []
        for data in (raw1, raw2):
            joints = data[:, :22*3].reshape(-1, 22, 3)
            mp_joint.append(joints)

        save_path = os.path.join(out_dir, fname.replace(".npy", ".mp4"))
        title = fname.replace(".npy", "")

        print(f"[渲染] {fname} → {os.path.basename(save_path)}")
        plot_3d_motion(
            save_path=save_path,
            kinematic_tree=t2m_kinematic_chain,
            mp_joints=mp_joint,
            title=title,
            fps=fps,
            radius=radius
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch visualize paired .npy motions into MP4, using your plot_3d_motion"
    )
    parser.add_argument("--npy_dir1", required=True, help="person1 npy 目录")
    parser.add_argument("--npy_dir2", required=True, help="person2 npy 目录")
    parser.add_argument("--out_dir",  required=True, help="输出 mp4 目录")
    parser.add_argument("--fps", type=int, default=30, help="输出视频帧率")
    parser.add_argument("--radius", type=float, default=4, help="渲染视图半径")
    args = parser.parse_args()

    visualize_pair(
        npy_dir1=args.npy_dir1,
        npy_dir2=args.npy_dir2,
        out_dir=args.out_dir,
        fps=args.fps,
        radius=args.radius
    )
