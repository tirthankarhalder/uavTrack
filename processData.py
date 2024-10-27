import json
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os

filePath = "./data/drone_sense_cse_base_1.txt"
data = []
with open(filePath,"r") as file:
    for line in file:
        data.append(json.loads(line)["answer"])

df = pd.DataFrame(data)
print(df.columns)
print(df.shape)

print("Dropping NAN rows")
df = df.dropna()


plotPath = "./visualization/test/"
os.makedirs(plotPath, exist_ok=True)

for index,row in df.iterrows():
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(111,projection='3d')
    frames=10
    img1 = ax1.scatter([e for elm in df['x_coord'][index:index+frames] for e in elm], [e for elm in df['y_coord'][index:index+frames] for e in elm], [e for elm in df['z_coord'][index:index+frames] for e in elm], cmap = [e for elm in df['dopplerIdx'][index:index+frames] for e in elm], marker='o')
    fig.colorbar(img1,ax=ax1, label='Intensity')
    ax1.set_title('Radar PCD')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-10,10)
    ax1.set_ylim(0,10)
    ax1.set_zlim(-3,10)
    plt.tight_layout()
    plt.savefig(f"{plotPath}/test_{str(index)}.png")
    # plt.show()
    plt.close()
    # break