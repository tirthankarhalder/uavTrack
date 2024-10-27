import json
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import DBSCAN


def calculate_centroids(data, labels):
    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        cluster_points = data[label == labels]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append((label, centroid))
    return centroids


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
    x_coords = [e for elm in df['x_coord'][index:index+frames] for e in elm]
    y_coords = [e for elm in df['y_coord'][index:index+frames] for e in elm]
    z_coords = [e for elm in df['z_coord'][index:index+frames] for e in elm]
    img1 = ax1.scatter(x_coords, y_coords, z_coords, cmap = [e for elm in df['dopplerIdx'][index:index+frames] for e in elm], marker='o')
    fig.colorbar(img1,ax=ax1, label='Intensity')
    ax1.set_title('Radar PCD')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-10,10)
    ax1.set_ylim(0,10)
    ax1.set_zlim(-3,10)
    data = np.array([x_coords, y_coords, z_coords]).T
    clustering = DBSCAN(eps=1, min_samples=5).fit(data)
    cluster_labels = np.array(clustering.labels_)
    previous_centroids = calculate_centroids(data, cluster_labels)
    print(previous_centroids[0][1])
    for elem in previous_centroids:
        ax1.scatter(elem[1][0],elem[1][1],elem[1][2],marker='o', s=200)
    print(previous_centroids)
    plt.tight_layout()
    plt.savefig(f"{plotPath}/test_{str(index)}.png")
    plt.close()
