#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python adjust_batch_concat_motion_pos_rot.py \
  --input_dir  /cvhci/temp/yyang/InterGen/data/motions_processed \
  --pool_dir   /cvhci/temp/yyang/InterGen/data/motions_concat_pool \
  --mapping    /cvhci/temp/yyang/InterGen/data/motions_concat_pool/mapping.txt \
  --output_dir /cvhci/temp/yyang/InterGen/data/motions_concat_result \
  --seed       42 \
  --blend      10 \
  --id_start   1400 \
  --id_end     1450

"""
import os
import argparse
import random
import numpy as np

def ensure_split(arr):
    """
    输入 arr (T, D), D == 66+126 == 192
    返回 pos (T,22,3) 和 rot (T,21,6)
    """
    T, D = arr.shape
    assert D >= 66+126, f"D={D} < 192"
    pos = arr[:, :66].reshape(T, 22, 3)   # (x,y,z) 且 y 是高度
    rot = arr[:, 66:66+126].reshape(T, 21, 6)
    return pos, rot

def merge_flat(pos, rot):
    """
    pos: (T,22,3), rot: (T,21,6) → flatten 回 (T,192)
    """
    T = pos.shape[0]
    return np.concatenate([pos.reshape(T,66), rot.reshape(T,126)], axis=1)

def concat_smooth_motion(a, b, blend=0):
    """
    对单人 motion a,b 做拼接平滑：
      - a,b: (T,192)
      - 1) 分离 pos, rot
      - 2) pos 根对齐 b 到 a; rot 不对齐
      - 3) 对 pos 与 rot 在接缝处都插入 blend 帧的线性过渡
      - 4) 返回 flatten 后的 (T',192)
    """
    posA, rotA = ensure_split(a)  # posA:(Ta,22,3), rotA:(Ta,21,6)
    posB, rotB = ensure_split(b)

    # 根对齐 posB → posA
    delta = posA[-1,0] - posB[0,0]  # pelvis offset
    posB_shift = posB + delta[None,None,:]

    # 直接拼或带过渡
    if blend <= 0:
        posC = np.concatenate([posA, posB_shift], axis=0)
        rotC = np.concatenate([rotA, rotB], axis=0)
    else:
        # 1) pos 平滑
        endA = posA[-1]      # (22,3)
        startB = posB_shift[0]
        pos_trans = []
        for i in range(1, blend+1):
            alpha = i/(blend+1)
            pos_trans.append((1-alpha)*endA + alpha*startB)
        pos_trans = np.stack(pos_trans,axis=0)  # (blend,22,3)
        posC = np.concatenate([posA[:-1], pos_trans, posB_shift], axis=0)

        # 2) rot 平滑
        endRA = rotA[-1]     # (21,6)
        startRB = rotB[0]
        rot_trans = []
        for i in range(1, blend+1):
            alpha = i/(blend+1)
            rot_trans.append((1-alpha)*endRA + alpha*startRB)
        rot_trans = np.stack(rot_trans,axis=0)  # (blend,21,6)
        rotC = np.concatenate([rotA[:-1], rot_trans, rotB], axis=0)

    # flatten 并返回
    return merge_flat(posC, rotC)

def load_mapping(path):
    """
    加载映射，每行 'pool_fname<空白>annotation_text'
    """
    mp = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(None,1)
            if len(parts)==2:
                mp[parts[0]] = parts[1]
    return mp

def main():
    p = argparse.ArgumentParser(
        description="批量拼接 6D-rot 动作并平滑过渡，生成同名注释 txt"
    )
    p.add_argument("--input_dir",   required=True,
                   help="原始 motions 根目录, 含 person1/ person2")
    p.add_argument("--pool_dir",    required=True,
                   help="拼接用 motion pool 根目录, 含 person1/ person2")
    p.add_argument("--mapping",     required=True,
                   help="映射文件: pool_fname<空白>annotation_text")
    p.add_argument("--output_dir",  required=True,
                   help="输出根目录, 生成 person1/ person2/ 及 txt")
    p.add_argument("--seed",        type=int, default=None,
                   help="随机种子(可选)")
    p.add_argument("--blend",       type=int, default=0,
                   help="接缝处插入线性过渡帧数")
    p.add_argument("--id_start",    type=int, required=True,
                   help="处理 motion 编号区间 起 (含)")
    p.add_argument("--id_end",      type=int, required=True,
                   help="处理 motion 编号区间 止 (不含)")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    mapping  = load_mapping(args.mapping)
    pool_list= sorted(f for f in os.listdir(os.path.join(args.pool_dir,"person1"))
                      if f.endswith(".npy"))
    if not pool_list:
        raise RuntimeError("pool_dir/person1 下无 .npy 文件")

    # 筛选原始 motion
    in1 = sorted(f for f in os.listdir(os.path.join(args.input_dir,"person1"))
                 if f.endswith(".npy"))
    in2 = sorted(f for f in os.listdir(os.path.join(args.input_dir,"person2"))
                 if f.endswith(".npy"))
    common=[]
    for fn in in1:
        try:
            idx = int(os.path.splitext(fn)[0])
        except:
            continue
        if args.id_start <= idx < args.id_end and fn in in2:
            common.append(fn)
    common.sort(key=lambda fn:int(fn[:-4]))
    print("待处理文件：", common)

    # 准备输出目录
    out1 = os.path.join(args.output_dir, "person1")
    out2 = os.path.join(args.output_dir, "person2")
    os.makedirs(out1, exist_ok=True)
    os.makedirs(out2, exist_ok=True)

    # 批量处理
    for fname in common:
        # 随机选 pool
        pool_fname = random.choice(pool_list)
        ann_text   = mapping.get(pool_fname, f"appended with {pool_fname}")

        # 载入
        o1 = np.load(os.path.join(args.input_dir,"person1", fname))
        o2 = np.load(os.path.join(args.input_dir,"person2", fname))
        p1 = np.load(os.path.join(args.pool_dir,"person1", pool_fname))
        p2 = np.load(os.path.join(args.pool_dir,"person2", pool_fname))

        # 拼接 + 平滑
        r1 = concat_smooth_motion(o1, p1, blend=args.blend)
        r2 = concat_smooth_motion(o2, p2, blend=args.blend)

        # 保存 npy
        np.save(os.path.join(out1, fname), r1)
        np.save(os.path.join(out2, fname), r2)
        # 写注释 txt
        txt_path = os.path.join(args.output_dir, fname.replace(".npy",".txt"))
        with open(txt_path, "w", encoding="utf-8") as ft:
            ft.write(ann_text + "\n")

        print(f"[OK] {fname} + {pool_fname} → shapes {r1.shape}, 注释→{txt_path}")

    print("全部完成，输出目录:", args.output_dir)

if __name__=="__main__":
    main()
