#!/usr/bin/env python

"""

  python adjust_detect_kick_interp2.py \
  --input_dir1 /cvhci/temp/yyang/InterGen/data/motions_processed/person1 \
  --input_dir2 /cvhci/temp/yyang/InterGen/data/motions_processed/person2 \
  --output_dir1 /cvhci/temp/yyang/InterGen/data/motions_repeat_kick/person1 \
  --output_dir2 /cvhci/temp/yyang/InterGen/data/motions_repeat_kick/person2 \
  --detect_person 1 \
  --mode repeat \
  --interp_blend 3 \
  --vel_thre 0.02 \
  --height_thre 0.3 \
  --start 1405 --end 1407

"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse
import numpy as np

def detect_segments(positions, vel_thre, height_thre, fid_l, fid_r):
    """
    在 (T,22,3) 的贴地关节位置上检测踢击帧段，
    返回连续踢击的 (start_idx, end_idx) 列表（闭区间）。
    """
    diffs   = positions[1:] - positions[:-1]      # (T-1,22,3)
    speeds  = np.linalg.norm(diffs, axis=2)       # (T-1,22)
    heights = positions[:-1,:,1]                  # (T-1,22)

    v_l = speeds[:, fid_l];  h_l = heights[:, fid_l]
    v_r = speeds[:, fid_r];  h_r = heights[:, fid_r]

    kick = (v_l > vel_thre) & (h_l > height_thre) \
         | (v_r > vel_thre) & (h_r > height_thre)

    segments = []
    Tm1 = positions.shape[0] - 1
    i = 0
    while i < Tm1:
        if kick[i]:
            start = i+1
            j = i+1
            while j < Tm1 and kick[j]:
                j += 1
            end = j
            segments.append((start, end))
            i = j
        else:
            i += 1

    # 排除首帧段 & 合并可能重叠
    segs = []
    for s,e in segments:
        if s>0:
            if not segs or s>segs[-1][1]:
                segs.append([s,e])
            else:
                # 重叠或相邻，合并
                segs[-1][1] = max(segs[-1][1], e)
    return [(s,e) for s,e in segs]

def process_pair_raw(p1, p2, args):
    r1 = np.load(p1); r2 = np.load(p2)
    T, D = r1.shape
    if r2.shape != (T,D):
        raise ValueError("形状不匹配")
    if D < 66:
        raise ValueError("维度不足")

    # pos1 = r1[:, :66].reshape(T,22,3)
    # pos2 = r2[:, :66].reshape(T,22,3)
    # 取前66维 → (T,22,3)，原始顺序是 (x,z,y)，要改成 (x,y,z), y轴为高度
    pos1 = r1[:,:66].reshape(T,22,3)[:,:, [0,2,1]]
    pos2 = r2[:,:66].reshape(T,22,3)[:,:, [0,2,1]]
    # # 贴地
    # floor_y = min(pos1[:,:,1].min(), pos2[:,:,1].min())
    # pos1[:,:,1] -= floor_y
    # pos2[:,:,1] -= floor_y
    # 分别贴地（只要检测者脚平面贴地即可）
    if args.detect_person == 1:
        floor_y = pos1[:,:,1].min()
    else:
        floor_y = pos2[:,:,1].min()
    pos1[:,:,1] -= floor_y
    pos2[:,:,1] -= floor_y

    # DEBUG：检查看贴地后最低值是不是 0
    print("贴地后最低高度：",
          pos1[:,:,1].min(), pos2[:,:,1].min())

    seq = pos1 if args.detect_person==1 else pos2
    segments = detect_segments(
        seq,
        args.vel_thre, args.height_thre,
        args.fid_l, args.fid_r
    )
    print(f"{os.path.basename(p1)} segments:", segments)
    if not segments:
        return r1, r2, []

    out1, out2 = [], []
    cur = 0
    for (s,e) in segments:
        # 1) 添加前段
        if cur < s:
            out1.append(r1[cur:s])
            out2.append(r2[cur:s])

        # 2) 添加踢击段：先原始，再重复
        segment1 = r1[s:e+1]
        segment2 = r2[s:e+1]
        # 原始一份
        out1.append(segment1)
        out2.append(segment2)
        if args.mode == "repeat":
            for _ in range(args.repeat_count):
                out1.append(segment1)
                out2.append(segment2)

        # 3) 插值过渡
        next_idx = e+1 if e+1 < T else e
        prev1 = segment1[-1]
        prev2 = segment2[-1]
        next1 = r1[next_idx]
        next2 = r2[next_idx]
        for i in range(args.interp_blend):
            alpha = (i+1)/(args.interp_blend+1)
            f1 = (1-alpha)*prev1 + alpha*next1
            f2 = (1-alpha)*prev2 + alpha*next2
            out1.append(f1[None])
            out2.append(f2[None])

        cur = e+1

    # 4) 添加尾段
    if cur < T:
        out1.append(r1[cur:])
        out2.append(r2[cur:])

    new1 = np.concatenate(out1, axis=0)
    new2 = np.concatenate(out2, axis=0)
    return new1, new2, segments

def main():
    p = argparse.ArgumentParser(description="批量踢击段删除/重复+插值过渡")
    p.add_argument("--input_dir1",  required=True)
    p.add_argument("--input_dir2",  required=True)
    p.add_argument("--output_dir1", required=True)
    p.add_argument("--output_dir2", required=True)
    p.add_argument("--detect_person", type=int, choices=[1,2], default=1)
    p.add_argument("--mode", choices=["remove","repeat"], default="remove")
    p.add_argument("--repeat_count", type=int, default=1,
                   help="额外重复次数")
    p.add_argument("--vel_thre",    type=float, default=0.02)
    p.add_argument("--height_thre", type=float, default=0.1)
    p.add_argument("--fid_l",       type=int, default=11)
    p.add_argument("--fid_r",       type=int, default=10)
    p.add_argument("--interp_blend", type=int, default=3,
                   help="踢击后过渡插值帧数")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end",   type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir1, exist_ok=True)
    os.makedirs(args.output_dir2, exist_ok=True)

    files1 = sorted(f for f in os.listdir(args.input_dir1) if f.endswith(".npy"))
    files2 = sorted(f for f in os.listdir(args.input_dir2) if f.endswith(".npy"))
    common = sorted(set(files1)&set(files2),
                    key=lambda fn:int(os.path.splitext(fn)[0]))
    common = common[args.start:args.end]

    print(f"处理 {len(common)} 个 motion → mode={args.mode}, blend={args.interp_blend}")
    for fn in common:
        p1 = os.path.join(args.input_dir1, fn)
        p2 = os.path.join(args.input_dir2, fn)
        try:
            n1, n2, segs = process_pair_raw(p1,p2,args)
            np.save(os.path.join(args.output_dir1, fn), n1)
            np.save(os.path.join(args.output_dir2, fn), n2)
            print(f"[OK] {fn}: segs={segs} → 新长度 {n1.shape[0]}")
        except Exception as e:
            print(f"[跳过] {fn}: {e}")

if __name__=="__main__":
    main()
