"""
python adjust_concat_motion.py \
  --input_dir /cvhci/temp/yyang/InterGen/data/motions_tobe_concat/person1 \
  --suffix /cvhci/temp/yyang/InterGen/data/motions_processed/person1/2.npy \
  --blend 10 \
  --output_dir /cvhci/temp/yyang/InterGen/data/motions_concat/person1

"""
import os
import argparse
import numpy as np

def ensure_xyz(arr):
    if arr.ndim == 2 and arr.shape[1] % 3 == 0:
        T, D = arr.shape
        return arr.reshape(T, D // 3, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"无法识别的数组形状 {arr.shape}")

def translate_and_blend(seqA, seqB, blend=0):
    """
    seqA, seqB: (TA,J,3), (TB,J,3)
    返回 (TA+TB-blend, J, 3) 或 (TA+TB, J,3)(blend=0)
    """
    rootA = seqA[-1, 0]
    rootB = seqB[0, 0]
    delta = rootA - rootB
    B_t = seqB + delta[None, None, :]

    if blend <= 0:
        return np.concatenate([seqA, B_t], axis=0)

    # 限制 blend 大小
    blend = min(blend, seqA.shape[0]-1, seqB.shape[0]-1)

    OUT = []
    OUT.append(seqA[:-blend])  # (TA-blend, J,3)

    # 插入过渡帧，每帧都扩成 (1, J,3)
    for i in range(blend):
        alpha = (i+1)/(blend+1)
        fA = seqA[-blend + i]          # (J,3)
        fB = B_t[i]                    # (J,3)
        mix = (1-alpha)*fA + alpha*fB          # (J,3)
        OUT.append(mix[None, ...])     # (1,J,3)

    OUT.append(B_t[blend:])           # (TB-blend, J,3)

    return np.concatenate(OUT, axis=0)

def main():
    parser = argparse.ArgumentParser(
        description="批量拼接：根对齐 + 可选过渡（已修复过渡帧维度）"
    )
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--suffix",     required=True)
    parser.add_argument("--blend",      type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seqB_raw = np.load(args.suffix)
    seqB = ensure_xyz(seqB_raw)

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
    print(f"共处理 {len(files)} 个文件，尾部：{os.path.basename(args.suffix)}")

    for fname in files:
        pathA = os.path.join(args.input_dir, fname)
        try:
            seqA_raw = np.load(pathA)
            seqA = ensure_xyz(seqA_raw)

            C_xyz = translate_and_blend(seqA, seqB, blend=args.blend)

            T, J, _ = C_xyz.shape
            C = C_xyz.reshape(T, J*3)

            out_path = os.path.join(args.output_dir, fname)
            np.save(out_path, C)
            print(f"[OK] {fname} → {C.shape}")

        except Exception as e:
            print(f"[跳过] {fname}: {e}")

if __name__ == "__main__":
    main()
