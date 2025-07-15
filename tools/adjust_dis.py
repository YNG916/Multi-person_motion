"""
python adjust_dis.py \
  --data_root /cvhci/temp/yyang/InterGen/data/motions_processed \
  --factor 1.5 \
  --sym
"""
import os, argparse
import numpy as np

def adjust_pair(p1_path, p2_path, factor=1.0, sym=False):
    """
    加载 person1/2 npy → reshape → 按首帧骨盆距缩放 → flatten → 返回新的 (T, D)
    """
    # 1) 加载
    p1 = np.load(p1_path)   # shape (T, 82*3)
    p2 = np.load(p2_path)

    # 2) 还原 (T, J, 3)
    T, D = p1.shape
    J = D // 3
    p1 = p1.reshape(T, J, 3)
    p2 = p2.reshape(T, J, 3)

    # 3) 取首帧骨盆(0号关节)位置
    root1 = p1[0, 0]    # (3,)
    root2 = p2[0, 0]    # (3,)

    # 4) 计算原始向量 & 新向量
    delta = root2 - root1           # 原始位移向量
    new_delta = delta * factor      # 按比例缩放
    shift = new_delta - delta       # 需要平移的增量

    # 5) 应用平移
    if sym:
        # 两人对称：各移动半个增量
        p1 = p1 - shift[None]/2
        p2 = p2 + shift[None]/2
    else:
        # 只移动第二个人
        p2 = p2 + shift[None]

    # 6) flatten 回 (T, J*3)
    p1_new = p1.reshape(T, J*3)
    p2_new = p2.reshape(T, J*3)
    return p1_new, p2_new

def main():
    parser = argparse.ArgumentParser(
        description="批量调整双人动作初始距离"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="motions_processed 根目录，包含 person1/ person2 子目录")
    parser.add_argument("--factor", type=float, default=1.0,
                        help="初始骨盆距离缩放倍数 (>1 拉远, <1 推近)")
    parser.add_argument("--sym", action="store_true",
                        help="是否两人对称移动（否则只移动第二人）")
    args = parser.parse_args()

    dir1 = os.path.join(args.data_root, "person1")
    dir2 = os.path.join(args.data_root, "person2")

    files1 = set(f for f in os.listdir(dir1) if f.endswith(".npy"))
    files2 = set(f for f in os.listdir(dir2) if f.endswith(".npy"))
    common = sorted(files1 & files2, key=lambda fn: int(os.path.splitext(fn)[0]))
    # 选择要转化的motions的序号区间
    common = common[1400:1450]

    for fname in common:
        p1_path = os.path.join(dir1, fname)
        p2_path = os.path.join(dir2, fname)
        try:
            p1, p2 = adjust_pair(p1_path, p2_path,
                                         factor=args.factor,
                                         sym=args.sym)
        except Exception as e:
            print(f"跳过 {fname}: {e}")
            continue

        # 保存
        BASE = os.path.dirname(os.path.abspath(__file__))
        BASE = os.path.dirname(BASE)
        out_dir = os.path.join(BASE, "data", "motions_change_distance")
        os.makedirs(os.path.join(out_dir, "person1"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "person2"), exist_ok=True)
        np.save(os.path.join(out_dir, "person1", fname), p1)
        np.save(os.path.join(out_dir, "person2", fname), p2)
        print(f"{fname} → new distance: factor={args.factor}, sym={args.sym}")

if __name__ == "__main__":
    main()
