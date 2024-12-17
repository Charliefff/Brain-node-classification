import torch
import matplotlib.pyplot as plt
import os

def plot_attention_heatmap(atten, epoch=0, title="Attention Heatmap", save_path="./attention_heatmap.png"):
    atten = atten.squeeze()  
    atten_2d = atten.unsqueeze(1)  
 
    plt.figure(figsize=(6, 8))  
    plt.imshow(atten_2d, cmap='bwr', aspect='auto')  
    plt.colorbar(label="Attention Weight")  
    plt.title(title)
    plt.ylabel("Attention Values")    
    
    plt.xticks([0], ['Feature'], fontsize=10)  
    plt.xlabel("")  
    
    for i in range(atten_2d.shape[0]):  
        for j in range(atten_2d.shape[1]):  
            value = atten_2d[i, j].item()  
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color="black", fontsize=8)

    if save_path:
        save_path = f"{save_path}_{epoch}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')  
        print(f"Heatmap saved to: {save_path}")

    plt.close() 

if __name__ == "__main__":
    
    atten = torch.randn(1, 1, 30, 1)  
    plot_attention_heatmap(atten, epoch=1, title="Attention Heatmap with Values", save_path="attention_pic/heatmap")
