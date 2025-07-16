#!/usr/bin/env python

"""

  python adjust_detect_kick_interp.py \
  --input_dir1 /cvhci/temp/yyang/InterGen/data/motions_processed/person1 \
  --input_dir2 /cvhci/temp/yyang/InterGen/data/motions_processed/person2 \
  --output_dir1 /cvhci/temp/yyang/InterGen/data/motions_delete_kick/person1 \
  --output_dir2 /cvhci/temp/yyang/InterGen/data/motions_delete_kick/person2 \
  --detect_person 1 \
  --mode remove \
  --interp_blend 5 \
  --vel_thre 0.02 \
  --height_thre 0.1 \
  --start 1405 --end 1407

"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np

def detect_segments(positions, vel_thre, height_thre, fid_l, fid_r):
    """
    在 (T,22,3) 的贴地关节位置上检测踢击帧段：
    返回一个列表，每项是 (start_idx, end_idx) 的闭区间，
    表示一段连续的踢击帧。
    """
    # 速度与高度
    diffs   = positions[1:] - positions[:-1]      # (T-1,22,3)
    speeds  = np.linalg.norm(diffs, axis=2)       # (T-1,22)
    heights = positions[:-1,:,1]                  # (T-1,22)

    v_l = speeds[:, fid_l];  h_l = heights[:, fid_l]
    v_r = speeds[:, fid_r];  h_r = heights[:, fid_r]

    kick = (v_l > vel_thre) & (h_l > height_thre) \
         | (v_r > vel_thre) & (h_r > height_thre)

    # 识别连续段
    segments = []
    i = 0
    Tm1 = positions.shape[0] - 1
    while i < Tm1:
        if kick[i]:
            start = i + 1
            j = i + 1
            # 扩展到最后一个 True
            while j < Tm1 and kick[j]:
                j += 1
            end = j  # 踢击发生在 diffs[j-1] 对应帧 j
            segments.append((start, end))
            i = j
        else:
            i += 1
    # 排除首帧
    segments = [(s,e) for (s,e) in segments if s > 0]
    return segments

def process_and_save_pair(p1, p2, args):
    """
    对一对原始 (T,492) npy:
      1) 载入 raw1, raw2
      2) 贴地 normalize(前66维)
      3) detect_segments → segments
      4) 在两人上同步执行 remove/repeat + 插值过渡
    返回 new1, new2, segments
    """
    r1 = np.load(p1); r2 = np.load(p2)
    T, D = r1.shape
    if r2.shape != (T, D):
        raise ValueError("两文件形状不匹配")
    if D < 66:
        raise ValueError("维度不足")

    # 前66维 reshape→(T,22,3), y为高度
    # pos1 = r1[:,:66].reshape(T,22,3)
    # pos2 = r2[:,:66].reshape(T,22,3)
    pos1 = r1[:,:66].reshape(T,22,3)[:,:, [0,2,1]]
    pos2 = r2[:,:66].reshape(T,22,3)[:,:, [0,2,1]]
    # floor normalize
    floor_y = min(pos1[:,:,1].min(), pos2[:,:,1].min())
    pos1[:,:,1] -= floor_y
    pos2[:,:,1] -= floor_y

    seq = pos1 if args.detect_person==1 else pos2
    segments = detect_segments(seq,
        args.vel_thre, args.height_thre,
        args.fid_l, args.fid_r
    )
    if not segments:
        return r1, r2, segments

    out1, out2 = [], []
    cur = 0
    for (s,e) in segments:
        # 1) 添加段前帧
        out1.append(r1[cur:s])
        out2.append(r2[cur:s])

        # 2) 根据模式添加踢击段
        segment1 = r1[s:e+1]
        segment2 = r2[s:e+1]
        if args.mode == "remove":
            # nothing
            pass
        else:  # repeat
            for _ in range(args.repeat_count):
                out1.append(segment1)
                out2.append(segment2)

        # 3) 插值过渡 frames
        # prev frame = last of out1[-1] if exists else r1[e+1]
        prev1 = out1[-1][-1] if len(out1[-1])>0 else r1[e+1]
        prev2 = out2[-1][-1] if len(out2[-1])>0 else r2[e+1]
        next1 = r1[e+1] if e+1 < T else prev1
        next2 = r2[e+1] if e+1 < T else prev2

        for i in range(args.interp_blend):
            alpha = (i+1)/(args.interp_blend+1)
            f1 = (1-alpha)*prev1 + alpha*next1
            f2 = (1-alpha)*prev2 + alpha*next2
            out1.append(f1[None])
            out2.append(f2[None])

        cur = e+1

    # 最后剩余尾部
    if cur < T:
        out1.append(r1[cur:])
        out2.append(r2[cur:])

    # 拼接所有块
    new1 = np.concatenate(out1, axis=0)
    new2 = np.concatenate(out2, axis=0)
    return new1, new2, segments

def main():
    p = argparse.ArgumentParser(
        description="批量删除/重复踢腿整段，并插值过渡"
    )
    p.add_argument("--input_dir1",  required=True)
    p.add_argument("--input_dir2",  required=True)
    p.add_argument("--output_dir1", required=True)
    p.add_argument("--output_dir2", required=True)
    p.add_argument("--detect_person", type=int, choices=[1,2], default=1)
    p.add_argument("--mode", choices=["remove","repeat"], default="remove")
    p.add_argument("--repeat_count", type=int, default=1)
    p.add_argument("--vel_thre",    type=float, default=0.02)
    p.add_argument("--height_thre", type=float, default=0.1)
    p.add_argument("--fid_l",       type=int, default=11)
    p.add_argument("--fid_r",       type=int, default=10)
    p.add_argument("--interp_blend", type=int, default=5,
        help="踢击段后插值过渡帧数"
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end",   type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir1, exist_ok=True)
    os.makedirs(args.output_dir2, exist_ok=True)

    files1 = {f for f in os.listdir(args.input_dir1) if f.endswith(".npy")}
    files2 = {f for f in os.listdir(args.input_dir2) if f.endswith(".npy")}
    common = sorted(files1&files2, key=lambda fn:int(fn[:-4]))
    common = common[args.start:args.end]
    print(f"处理 {len(common)} 个 motion，mode={args.mode}, blend={args.interp_blend}")

    for fn in common:
        p1 = os.path.join(args.input_dir1, fn)
        p2 = os.path.join(args.input_dir2, fn)
        try:
            n1, n2, segs = process_and_save_pair(p1,p2,args)
            np.save(os.path.join(args.output_dir1,fn), n1)
            np.save(os.path.join(args.output_dir2,fn), n2)
            print(f"{fn}: 删除/重复段 {segs} → 新长度 {n1.shape[0]}")
        except Exception as e:
            print(f"跳过 {fn}: {e}")

if __name__=="__main__":
    main()

