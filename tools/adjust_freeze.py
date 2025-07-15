"""
python adjust_freeze.py \
  --data_root /cvhci/temp/yyang/InterGen/data/motions_processed \
  --freeze 2 
"""
import os
import argparse
import numpy as np

def freeze_pair(p1_path, p2_path, freeze_person=1):
    """
      - freeze → 所有帧都等于第一帧
      - other  → 保持不变
    """
    p1 = np.load(p1_path)  # (T, D)
    p2 = np.load(p2_path)
    T, D = p1.shape

    if freeze_person == 1:
        # person1 冷冻
        p1_new = np.tile(p1[0:1, :], (T, 1))
        p2_new = p2.copy()
    else:
        # person2 冷冻
        p1_new = p1.copy()
        p2_new = np.tile(p2[0:1, :], (T, 1))

    return p1_new, p2_new

def main():
    parser = argparse.ArgumentParser(
        description="批量冻住双人动作中的某一位，只保留另一个人的原动作"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="motions_processed 根目录，包含 person1/ person2 子目录"
    )
    parser.add_argument(
        "--freeze", type=int, choices=[1,2], default=1,
        help="选择要冻结的那一人,1 或 2"
    )

    args = parser.parse_args()

    dir1 = os.path.join(args.data_root, "person1")
    dir2 = os.path.join(args.data_root, "person2")
    files1 = {f for f in os.listdir(dir1) if f.endswith(".npy")}
    files2 = {f for f in os.listdir(dir2) if f.endswith(".npy")}
    # 按数字前缀自然序
    common = sorted(
        files1 & files2,
        key=lambda fn: int(os.path.splitext(fn)[0])
    )
    # 选择motion序号
    common = common[1400:1450]

    for fname in common:
        p1_path = os.path.join(dir1, fname)
        p2_path = os.path.join(dir2, fname)
        try:
            p1_new, p2_new = freeze_pair(p1_path, p2_path, freeze_person=args.freeze)
        except Exception as e:
            print("跳过", fname, ":", e)
            continue

        # 保存
        # 保存
        BASE = os.path.dirname(os.path.abspath(__file__))
        BASE = os.path.dirname(BASE)
        out_dir = os.path.join(BASE, "data", "motions_freeze")
        os.makedirs(os.path.join(out_dir, "person1"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "person2"), exist_ok=True)
        np.save(os.path.join(out_dir, "person1", fname), p1_new)
        np.save(os.path.join(out_dir, "person2", fname), p2_new)

        print(f"处理完：{fname} → 冻结 person{args.freeze}，结果 shape {p1_new.shape}")

if __name__ == "__main__":
    main()
