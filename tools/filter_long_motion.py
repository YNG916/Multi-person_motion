#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# ———— 配置项 ————
# 请将 ROOT_DIR 修改为你存放 .npy 动作文件的根目录
ROOT_DIR = "/cvhci/temp/yyang/InterGen/data/motions_processed/person1"

# 输出的 Excel 名称
OUTPUT_EXCEL = "short_motions.xlsx"

# 判断帧数的阈值
FRAME_THRESHOLD = 600
# —————————————————


def find_short_motions(root_dir, frame_thresh):
    """
    递归遍历 root_dir 下所有 .npy 文件，筛选出帧数 < frame_thresh 的文件。
    返回一个列表，其中每个元素为 (相对路径, 帧数)。
    """
    short_list = []

    # 遍历 root_dir 下所有子目录和文件
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".npy"):
                continue

            full_path = os.path.join(dirpath, fname)
            try:
                arr = np.load(full_path)
            except Exception as e:
                print(f"无法加载文件 {full_path}，跳过。错误信息：{e}")
                continue

            # 如果是扁平化 (T, D)，直接取第一维；如果 (T, J, 3)，同样取第一维
            if arr.ndim >= 1:
                frame_count = arr.shape[0]
            else:
                # 如果数据维度异常，则跳过
                print(f"文件 {full_path} 不是有效的动作数组（维度={arr.ndim}），跳过。")
                continue

            if frame_count < frame_thresh:
                # 记录相对路径（相对于 root_dir）和帧数
                rel_path = os.path.relpath(full_path, root_dir)
                short_list.append((rel_path, frame_count))

    return short_list


def save_to_excel(short_list, output_excel):
    """
    将筛选结果写入 Excel，列名为 “编号”、“文件路径”、“帧数”。
    short_list 中的元素格式为 (相对路径, 帧数)。
    """
    # 按编号排序（其实 find_short_motions 已经是按遍历顺序，但这里再编号）
    data = {
        "编号": [],
        "文件路径": [],
        "帧数": []
    }
    for idx, (rel_path, frame_count) in enumerate(short_list, start=1):
        data["编号"].append(idx)
        data["文件路径"].append(rel_path)
        data["帧数"].append(frame_count)

    df = pd.DataFrame(data)
    # 将 DataFrame 写入 Excel（pandas 自动安装 openpyxl 或 xlwt）
    df.to_excel(output_excel, index=False)
    print(f"已将 {len(short_list)} 条记录保存到 {output_excel}")


if __name__ == "__main__":
    # 1. 筛选出所有帧数 < FRAME_THRESHOLD 的 .npy 文件
    short_motions = find_short_motions(ROOT_DIR, FRAME_THRESHOLD)

    # 2. 保存到 Excel
    save_to_excel(short_motions, OUTPUT_EXCEL)
