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

# T2M 的 22 关节骨骼树示例：
# kinematic_tree = [
#     [0,1],[1,2],[2,3],[3,4],       # 右腿
#     [0,5],[5,6],[6,7],[7,8],       # 左腿
#     [0,9],[9,10],[10,11],          # 躯干
#     [11,12],[12,13],               # 头
#     [10,14],[14,15],[15,16],       # 左臂
#     [10,17],[17,18],[18,19],       # 右臂
#     [8,20],[4,21]                  # 脚
# ]

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
    root = "/cvhci/temp/yyang/InterGen/data/motions_processed"
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

        seq1 = ensure_xyz(seq1)
        seq2 = ensure_xyz(seq2)

        seq1 = seq1[:, :, [0, 2, 1]]
        seq2 = seq2[:, :, [0, 2, 1]]

        # for seq in (seq1, seq2):
        #     root_x, root_z = seq[0, 0, 0], seq[0, 0, 2]
        #     seq[:, :, 0] -= root_x
        #     seq[:, :, 2] -= root_z
        for seq in (seq1, seq2):
            y_min = seq[:,:,1].min()
            seq[:,:,1] -= y_min

        save_path = os.path.join("/cvhci/temp/yyang/InterGen/visualize_results5", fname.replace(".npy", ".mp4"))
        title     = fname.replace(".npy", "")
        plot_3d_motion(save_path, kinematic_tree, [seq1, seq2],
                       title=title, fps=20)
