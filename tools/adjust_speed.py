import os, sys
import numpy as np
# speed.py 位于项目的 tools/ 子目录下，utils/ 与 tools/ 同级
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from utils.utils import process_motion_np, rigid_transform
from utils.quaternion import qmul_np, qinv_np, qrot_np
from utils.preprocess import load_motion

def resample_motion(data, alpha):
    """
    对 (T,192) 数据做时间重采样
    alpha>1→加速, alpha<1→减速
    """
    T, D = data.shape
    orig_t = np.arange(T)
    new_T = int(np.floor(T / alpha))
    new_t  = np.linspace(0, T-1, new_T) * alpha
    out = np.zeros((new_T, D), dtype=data.dtype)
    for i in range(D):
        out[:, i] = np.interp(new_t, orig_t, data[:, i])
    return out

def speed_and_align(p1_npy, p2_npy, alpha=2.0, min_length=1, n_joints=22):
    # # 1) 先从原始 .npy 文件加载并裁剪到 (T,192)
    # motion1, _ = load_motion(p1_npy, min_length, swap=False)  # -> (T,192)
    # motion2, _ = load_motion(p2_npy, min_length, swap=False)  # -> (T,192)
    # if motion1 is None or motion2 is None:
    #     raise ValueError("序列过短，无法处理", p1_npy, p2_npy)

    motion1 = np.load(p1_npy).astype(np.float32)  # e.g. (239, 246) or (239,192) after load_motion
    motion2 = np.load(p2_npy).astype(np.float32)

    # 2) 对 person1 做加速/减速重采样
    m1_fast = resample_motion(motion1, alpha)

    # 3) 保持 person2 同步长度（截断）
    # L = min(len(m1_fast), motion2.shape[0])
    L = min(len(m1_fast), len(motion2))
    m1_fast = m1_fast[:L]
    m2       = motion2[:L]

    return m1_fast, m2

    # # 4) 根节点对齐
    # m1_aligned, q1, pos1 = process_motion_np(m1_fast, feet_thre=0.001, prev_frames=0, n_joints=n_joints)
    # m2_aligned, q2, pos2 = process_motion_np(m2,      feet_thre=0.001, prev_frames=0, n_joints=n_joints)

    # # 5) 计算二者初始帧的相对旋转和平移
    # r_rel   = qmul_np(q2, qinv_np(q1))                   # (T,4)
    # angle   = np.arctan2(r_rel[:,2:3], r_rel[:,0:1])     # (T,1)
    # xz      = qrot_np(q1, pos2 - pos1)[:, [0,2]]         # (T,2)
    # relative= np.concatenate([angle, xz], axis=-1)       # (T,3)

    # # 6) 用首帧的相对刚体变换来摆正第二人
    # m2_final = rigid_transform(relative[0], m2_aligned)  # (T,192)

    # return m1_aligned, m2_final

if __name__ == "__main__":
    # 项目根目录
    BASE = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.dirname(BASE)
    DATA = os.path.join(BASE, "data", "motions_processed")

    dir1 = os.path.join(DATA, "person1")
    dir2 = os.path.join(DATA, "person2")
    common = sorted(set(os.listdir(dir1)) & set(os.listdir(dir2)))
    # 选择要转化的motions的序号区间
    common = sorted(common, key=lambda fn: int(os.path.splitext(fn)[0]))
    common = common[1400:1450]

    for fname in common:
        if not fname.endswith(".npy"):
            continue
        p1 = os.path.join(dir1, fname)
        p2 = os.path.join(dir2, fname)

        try:
            m1, m2 = speed_and_align(p1, p2, alpha=1.3)
        except ValueError as e:
            print("跳过", fname, e)
            continue

        # 保存加速并对齐后的 npy
        out_dir = os.path.join(BASE, "data", "motions_change_speed")
        os.makedirs(os.path.join(out_dir, "person1"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "person2"), exist_ok=True)
        np.save(os.path.join(out_dir, "person1", fname), m1)
        np.save(os.path.join(out_dir, "person2", fname), m2)
        print("处理完：", fname, "→", m1.shape)
