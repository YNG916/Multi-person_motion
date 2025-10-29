#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
python adjust_batch_concat_motion.py \
  --input_dir  /cvhci/temp/yyang/InterGen/data/motions_processed \
  --pool_dir   /cvhci/temp/yyang/InterGen/data/motions_concat_pool \
  --mapping    /cvhci/temp/yyang/InterGen/data/motions_concat_pool/mapping.txt \
  --output_dir /cvhci/temp/yyang/InterGen/data/motions_concat_result2 \
  --seed       42 \
  --blend      10 \
  --start      1400 \
  --end        1450
"""

import os
import argparse
import random
import numpy as np

# def ensure_xyz(arr):
#     """
#     将 (T, D) 或 (T, J, 3) 规范成 (T, J, 3)。
#     D 必须是 3 的倍数。
#     """
#     if arr.ndim == 2 and arr.shape[1] % 3 == 0:
#         T, D = arr.shape
#         return arr.reshape(T, D // 3, 3)
#     if arr.ndim == 3 and arr.shape[2] == 3:
#         return arr
#     raise ValueError(f"无法识别数组形状 {arr.shape}")

def concat_smooth(a, b, blend=0):
    """
    在末帧根对齐后拼接 a 与 b,并在接缝插入线性过渡。
    a, b: (Ta, J, 3), (Tb, J, 3)
    blend: 插入过渡帧数
    返回 (Ta + Tb + blend - 1, J, 3)  或 (Ta + Tb, J, 3)(blend=0)
    """
    # 1) 根对齐 b 到 a
    rootA = a[-1, 0]
    rootB = b[0, 0]
    delta = rootA - rootB
    b_shift = b + delta[None, None, :]

    if blend <= 0:
        # 直接拼接，保留 a 的所有帧
        return np.concatenate([a, b_shift], axis=0)

    # 2) 生成过渡帧，线性插值 a[-1] → b_shift[0]
    endA = a[-1]         # (J, 3)
    startB = b_shift[0]  # (J, 3)
    trans = []
    for i in range(1, blend + 1):
        alpha = i / (blend + 1)
        trans.append((1 - alpha) * endA + alpha * startB)
    trans = np.stack(trans, axis=0)  # (blend, J, 3)

    # 3) 拼接：去掉 a 的最后一帧（避免重复），再加上过渡帧和 b_shift
    return np.concatenate([a[:-1], trans, b_shift], axis=0)

def load_mapping(path):
    """
    从 mapping.txt 加载映射，每行格式：
      pool_fname <TAB> annotation_text, 
      e.g. 1.npy	then the person transitions into a walking motion
    返回字典 { pool_fname: annotation_text }
    """
    mp = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            # parts = line.strip().split('\t', 1)
            # if len(parts) == 2:
            #     mp[parts[0]] = parts[1]
            
            # split on any whitespace, maxsplit=1
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                key, text = parts
                mp[key] = text
    return mp

def main():
    parser = argparse.ArgumentParser(
        description="批量拼接基础 motion 并插入平滑过渡，每个输出生成同名 txt 注释"
    )
    parser.add_argument("--input_dir",  required=True,
                        help="原始 motions 根目录，含 person1/ person2 子目录")
    parser.add_argument("--pool_dir",   required=True,
                        help="基础 motion pool 根目录，含 person1/ person2 子目录")
    parser.add_argument("--mapping",    required=True,
                        help="映射文件，每行 pool_fname<TAB>annotation_text")
    parser.add_argument("--output_dir", required=True,
                        help="输出根目录，会生成 person1/ person2/ 及 txt 注释")
    parser.add_argument("--seed",       type=int, default=None,
                        help="随机种子，使结果可复现")
    parser.add_argument("--blend",      type=int, default=0,
                        help="在接缝处插入多少帧线性插值平滑")
    parser.add_argument("--start",      type=int, default=0,
                        help="原始 motions 列表切片起始索引（含）")
    parser.add_argument("--end",        type=int, default=None,
                        help="原始 motions 列表切片结束索引（不含）")
    args = parser.parse_args()

    # 可选随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 加载 annotation 映射
    mapping = load_mapping(args.mapping)
    
    p1_dir = os.path.join(args.pool_dir, "person1")
    print("Pool-dir person1 contains:", os.listdir(p1_dir))
    print("Mapping keys:", list(mapping.keys()))

    pool_files = sorted(
        f for f in os.listdir(os.path.join(args.pool_dir, "person1"))
        if f.endswith(".npy") and f in mapping
    )

    # 原始 motions 列表 & 切片
    in1 = sorted(f for f in os.listdir(os.path.join(args.input_dir, "person1"))
                 if f.endswith(".npy"))
    in2 = sorted(f for f in os.listdir(os.path.join(args.input_dir, "person2"))
                 if f.endswith(".npy"))
    # common = [f for f in in1 if f in in2][args.start:args.end]
    common = []
    for fn in in1:
        try:
            idx = int(os.path.splitext(fn)[0])
        except ValueError:
            continue
        if args.start <= idx < args.end and fn in in2:
            common.append(fn)
    common.sort(key=lambda fn: int(os.path.splitext(fn)[0]))


    # 输出目录
    out1 = os.path.join(args.output_dir, "person1")
    out2 = os.path.join(args.output_dir, "person2")
    os.makedirs(out1, exist_ok=True)
    os.makedirs(out2, exist_ok=True)

    for fname in common:
        # 随机选基础 motion
        pool_fname = random.choice(pool_files)
        ann_text   = mapping[pool_fname]

        # 加载原始 & 基础 motion
        o1 = np.load(os.path.join(args.input_dir,  "person1", fname))
        o2 = np.load(os.path.join(args.input_dir,  "person2", fname))
        p1 = np.load(os.path.join(args.pool_dir,   "person1", pool_fname))
        p2 = np.load(os.path.join(args.pool_dir,   "person2", pool_fname))

        # 确保 (T,J,3)，并把原始 (x,z,y) → (x,y,z)
        # a1 = ensure_xyz(o1)[:, :, [0, 2, 1]]
        # a2 = ensure_xyz(o2)[:, :, [0, 2, 1]]
        # b1 = ensure_xyz(p1)[:, :, [0, 2, 1]]
        # b2 = ensure_xyz(p2)[:, :, [0, 2, 1]]
        T1, D1 = o1.shape
        T2, D2 = p1.shape
        a1 = o1[:,:66].reshape(T1,22,3)[:, :, [0, 2, 1]]
        a2 = o2[:,:66].reshape(T1,22,3)[:, :, [0, 2, 1]]
        b1 = p1[:,:66].reshape(T2,22,3)[:, :, [0, 2, 1]]
        b2 = p2[:,:66].reshape(T2,22,3)[:, :, [0, 2, 1]]

        # 拼接并平滑
        r1 = concat_smooth(a1, b1, blend=args.blend)
        r2 = concat_smooth(a2, b2, blend=args.blend)

        # flatten 回 (T', J*3)
        TN1, J, _ = r1.shape
        TN2, _, _ = r2.shape
        # 变回 xzy
        r1 = r1[:, :, [0, 2, 1]]
        r2 = r2[:, :, [0, 2, 1]]
        out_arr1 = r1.reshape(TN1, J * 3)
        out_arr2 = r2.reshape(TN2, J * 3)

        # 保存 .npy
        np.save(os.path.join(out1, fname), out_arr1)
        np.save(os.path.join(out2, fname), out_arr2)
        # 保存同名 .txt 注释
        txt_path = os.path.join(args.output_dir, fname.replace(".npy", ".txt"))
        with open(txt_path, "w", encoding="utf-8") as ft:
            ft.write(ann_text + "\n")

        print(f"[OK] {fname} + {pool_fname} → shapes: {out_arr1.shape}, {out_arr2.shape}")

    print("All done. Outputs in", args.output_dir)

if __name__ == "__main__":
    main()

