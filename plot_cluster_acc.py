import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_category(base_dir, category, save_path):
    """绘制某一类的 cluster_acc 平均曲线与 std 阴影"""
    all_steps = None
    all_accs = []

    for i in range(1, 6):
        file_path = os.path.join(base_dir, f"{category}{i}", "encoding_monitor", "monitoring_data.json")
        if not os.path.exists(file_path):
            print(f"文件缺失: {file_path}")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)

        steps = np.array(data["steps"])
        acc = np.array(data["cluster_acc"])

        if all_steps is None:
            all_steps = steps
        else:
            # 确保 steps 对齐
            assert np.array_equal(all_steps, steps), f"{category}{i} 的 steps 不一致"

        all_accs.append(acc)

    all_accs = np.array(all_accs)
    mean_acc = np.mean(all_accs, axis=0)
    std_acc = np.std(all_accs, axis=0)

    plt.figure(figsize=(6, 4))
    plt.plot(all_steps, mean_acc, color="blue")
    plt.fill_between(all_steps, mean_acc - std_acc, mean_acc + std_acc, color="blue", alpha=0.2)
    plt.xlabel("steps")
    plt.ylabel("cluster_acc")
    plt.ylim(0, 1)  # 🔹 y轴固定在0-1
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存图像: {save_path}")

if __name__ == "__main__":
    base_dir = r"E:\Desktop\concept_discovery\tdmpc2\tdmpc2\logs\counting4\2022"
    categories = ["dense", "NP", "PJEPA", "random"]

    for cat in categories:
        save_path = os.path.join(base_dir, f"{cat.lower()}.png")
        plot_category(base_dir, cat, save_path)