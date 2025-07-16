import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3

# 导入骨骼树 
BASE = os.path.dirname(os.path.abspath(__file__))     # …/InterGen/tools
BASE = os.path.dirname(BASE)                          # …/InterGen
sys.path.insert(0, BASE) 
from utils.paramUtil import t2m_kinematic_chain as kinematic_tree
"""
t2m_kinematic_chain = [
    [0, 2,  5,  8, 11],      # 左下肢 pelvis → 左髋 → 左膝 → 左踝 → 左脚  
    [0, 1,  4,  7, 10],      # 右下肢 pelvis → 右髋 → 右膝 → 右踝 → 右脚  
    [0, 3,  6,  9, 12, 15],  # 躯干／头部 pelvis → 脊柱段1 → 脊柱段2 → 脊柱段3 → 颈部 → 头顶  
    [9, 14, 17, 19, 21],     # 左上肢 neck/chest → 左肩 → 左肘 → 左腕 → 左手  
    [9, 13, 16, 18, 20]      # 右上肢 neck/chest → 右肩 → 右肘 → 右腕 → 右手  
]
根节点index_0通常是骨盆(pelvis)。
每个数字:对应第N个关节在扁平化 (T, 22*3) 数组里的关节编号。
例如第一条链 [0,2,5,8,11]，你就按顺序把关节 0→2→5→8→11 的坐标依次用线画出来，得到左腿的 3D 骨架。
这样，只要给出一个 (T,22,3) 的关节坐标张量，并沿着 t2m_kinematic_chain 里的每条索引链去连线，就能得到整个骨骼按正确的拓扑结构。
mp_joints只取前22个joints的position, 即22*3=66维度

e.g.
def plot_t2m(self, mp_data, result_path, caption):
        mp_joint = []
        for i, data in enumerate(mp_data):
            if i == 0:
                joint = data[:,:22*3].reshape(-1,22,3)
            else:
                joint = data[:,:22*3].reshape(-1,22,3)

            mp_joint.append(joint)

        plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)
"""
def plot_3d_motion(save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=30, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])
    print(frame_number)

    # colors = ['red', 'blue', 'black', 'red', 'blue',
    #           'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
    #           'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    #
    colors = ['red', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    mp_offset = list(range(-len(mp_joints)//2, len(mp_joints)//2, 1))
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    for i,joints in enumerate(mp_joints):

        # (seq_len, joints_num, 3)
        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)


        #     print(data.shape)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        # data[:, :, 0] -= data[0:1, 0:1, 0]
        # data[:, :, 0] += mp_offset[i]
        #
        # data[:, :, 2] -= data[0:1, 0:1, 2]
        mp_data.append({"joints":data,
                        "MINS":MINS,
                        "MAXS":MAXS,
                        "trajec":trajec, })

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 15#7.5
        #         ax =
        plot_xzPlane(-3, 3, 0, -3, 3)
        for pid,data in enumerate(mp_data):
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                #             print(color)
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 1.0
                ax.plot3D(data["joints"][index, chain, 0], data["joints"][index, chain, 1], data["joints"][index, chain, 2], linewidth=linewidth,
                          color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()



if __name__ == "__main__":
    # root = "/cvhci/temp/yyang/InterGen/data/motions_processed"
    # root = "/cvhci/temp/yyang/InterGen/data/motions_change_speed"
    # root = "/cvhci/temp/yyang/InterGen/data/motions_change_distance"
    # root = "/cvhci/temp/yyang/InterGen/data/motions_freeze"
    # root = "/cvhci/temp/yyang/InterGen/data/motions_concat"
    root = "/cvhci/temp/yyang/InterGen/data/motions_delete_kick"
    # root = "/cvhci/temp/yyang/InterGen/data/motions_repeat_kick"




    dir1 = os.path.join(root, "person1")
    dir2 = os.path.join(root, "person2")

    # 只保留两边都有的 .npy 文件
    files1 = {f for f in os.listdir(dir1) if f.endswith(".npy")}
    files2 = {f for f in os.listdir(dir2) if f.endswith(".npy")}
    common = sorted(files1 & files2)


    for fname in common:
        path1 = os.path.join(dir1, fname)
        path2 = os.path.join(dir2, fname)

        seq1 = np.load(path1)
        seq2 = np.load(path2)

        # 如果是扁平化 (T, D)，就 reshape 成 (T, J, 3) —— 
        def ensure_xyz(arr):
            if arr.ndim == 2 and arr.shape[1] % 3 == 0:
                T, D = arr.shape
                J = D // 3
                return arr.reshape(T, J, 3)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                return arr
            else:
                raise ValueError(f"无法识别的数组形状 {arr.shape}")

        # seq1 = ensure_xyz(seq1)
        # seq2 = ensure_xyz(seq2)
         # 假设可视化 22 关节
        T, D = seq1.shape
        if D % 3 == 0 and D // 3 == 82:
            # 原始 82 markers
            J = 82
        else:
            # 简化 22 关节
            J = 22

        # 取前 J*3 列，重塑成 (T, J, 3)
        seq1 = seq1[:, :J*3].reshape(T, J, 3)
        seq2 = seq2[:, :J*3].reshape(T, J, 3)

        seq1 = seq1[:, :, [0, 2, 1]]
        seq2 = seq2[:, :, [0, 2, 1]]

        # for seq in (seq1, seq2):
        #     root_x, root_z = seq[0, 0, 0], seq[0, 0, 2]
        #     seq[:, :, 0] -= root_x
        #     seq[:, :, 2] -= root_z
        for seq in (seq1, seq2):
            y_min = seq[:,:,1].min()
            seq[:,:,1] -= y_min

        # save_path = os.path.join("/cvhci/temp/yyang/InterGen/visualize_results_change_speed", fname.replace(".npy", ".mp4"))
        # save_path = os.path.join("/cvhci/temp/yyang/InterGen/visualize_results_change_distance", fname.replace(".npy", ".mp4"))
        # save_path = os.path.join("/cvhci/temp/yyang/InterGen/visualize_results_freeze_person2", fname.replace(".npy", ".mp4"))
        # save_path = os.path.join("/cvhci/temp/yyang/InterGen/visualize_results_concat", fname.replace(".npy", ".mp4"))
        save_path = os.path.join("/cvhci/temp/yyang/InterGen/visualize_results_delete_kick", fname.replace(".npy", ".mp4"))
        # save_path = os.path.join("/cvhci/temp/yyang/InterGen/visualize_results_repeat_kick", fname.replace(".npy", ".mp4"))




        title     = fname.replace(".npy", "")
        plot_3d_motion(save_path, kinematic_tree, [seq1, seq2],
                       title=title, fps=20)
