import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def plot_attention_heatmap(atten, epoch=0, title="Attention Heatmap", save_path="./attention_heatmap.png"):
    atten = atten.squeeze()  
    atten_2d = atten.unsqueeze(1)
    
    # 自定義顏色映射：數值 < 0.3 設為白色
    cmap = plt.cm.bwr  # 原始顏色映射
    norm = mcolors.Normalize(vmin=atten_2d.min(), vmax=atten_2d.max())  # 正規化數值範圍
    colors = cmap(norm(atten_2d.numpy()))
    colors[abs(atten_2d.numpy()) < 0.03] = [1, 1, 1, 1]  # 小於 0.3 的設為白色 (RGBA: 1,1,1,1)

    # 繪製熱力圖
    plt.figure(figsize=(6, 8))
    plt.imshow(colors, aspect='auto')  # 使用修改後的顏色數據
    
    plt.title(title)
    plt.ylabel("Attention Values")

    plt.xticks([0], ['Feature'], fontsize=10)
    plt.xlabel("")
    
    
    for i in range(atten_2d.shape[0]):
        for j in range(atten_2d.shape[1]):
            value = atten_2d[i, j].item()
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color="black", fontsize=8)
    
    
    plt.gca().invert_yaxis()

    
    if save_path:
        save_path = f"{save_path}_{epoch}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")

    plt.close()

if __name__ == "__main__":
    # 生成原始注意力數據
    atten = torch.rand(1, 1, 30, 1)  # 生成 [0, 1] 之間的數據
    plot_attention_heatmap(atten, epoch=1, title="Attention Heatmap (<0.3 is White)", save_path="attention_pic/heatmap")
