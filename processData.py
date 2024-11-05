import json
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import DBSCAN
from matplotlib import cm
from scipy.stats import mode

def calculate_centroids(data, labels):
    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        cluster_points = data[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append((label, centroid))
    return centroids

def find_closest_centroid(previous_centroids, current_centroids, id_mapping, next_id, frame_counts, frame_threshold=20):
    new_centroids = []
    for curr_label, curr_centroid in current_centroids:
        min_distance = float('inf')
        closest_centroid = None
        closest_prev_label = None

        for prev_label, prev_centroid in previous_centroids:
            distance = np.linalg.norm(curr_centroid - prev_centroid)
            if distance < min_distance:
                min_distance = distance
                closest_centroid = curr_centroid
                closest_prev_label = prev_label

        if closest_prev_label is not None and min_distance < 0.5:  # Threshold distance
            # Use existing ID if close to a previous centroid, initializing if necessary
            centroid_id = id_mapping.get(closest_prev_label, next_id)
            if closest_prev_label not in id_mapping:
                id_mapping[closest_prev_label] = centroid_id
                frame_counts[centroid_id] = 0  # Initialize frame count if first occurrence
            frame_counts[centroid_id] += 1  # Increment frame count
        else:
            # Assign a new ID if no close match found
            centroid_id = next_id
            id_mapping[curr_label] = centroid_id
            frame_counts[centroid_id] = 1  # Initialize frame count
            next_id += 1

        # Store the ID in the mapping and in the new_centroids list
        new_centroids.append((centroid_id, curr_centroid))

    # Filter centroids based on frame threshold
    valid_centroids = [(cid, c) for cid, c in new_centroids if frame_counts[cid] >= frame_threshold]
    return valid_centroids, next_id

filePath = "./data/drone_data_2.txt"
visFile = filePath.split("/")[-1].split(".")[0]
data = []
with open(filePath, "r") as file:
    for line in file:
        data.append(json.loads(line)["answer"])

df = pd.DataFrame(data)
print(df.columns)
print(df.shape)

print("Dropping NAN rows")
df = df.dropna()
print(df.head(2))
newDF = df
for index, row in df.iterrows():
    # print(type(sum(df["dopplerIdx"][index])))
    if sum(df["dopplerIdx"][index]) == 0:
        # print(sum(df["dopplerIdx"][index]))
        df.drop(index=index)
        # print(f"index {index} dropped")
    
print("df.shape: ",df.shape)


plotPath = "./visualization/" + visFile
os.makedirs(plotPath, exist_ok=True)

previous_centroids = []
id_mapping = {}
frame_counts = {}
next_id = 0

# Define a color map for unique IDs
cmap = cm.get_cmap("tab20", 20)


for index, row in df.iterrows():
    frames = 10
    # Add break condition if there aren't enough frames left to process
    if index >= len(df) - frames:
        break
    print('index:', index)
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    x_coords = [e for elm in df['x_coord'][index:index + frames] for e in elm]
    y_coords = [e for elm in df['y_coord'][index:index + frames] for e in elm]
    z_coords = [e for elm in df['z_coord'][index:index + frames] for e in elm]
    doppler_idx = [e for elm in df['dopplerIdx'][index:index + frames] for e in elm]
    # print(doppler_idx)
    # break
    colors = [cmap(val % 20) for val in doppler_idx]
    
    img1 = ax1.scatter(x_coords, y_coords, z_coords, c=colors, marker='o')
    fig.colorbar(img1, ax=ax1, label='Intensity')
    ax1.set_title('Radar PCD')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(0, 10)
    ax1.set_zlim(-3, 10)

    data = np.array([x_coords, y_coords, z_coords]).T
    clustering = DBSCAN(eps=1, min_samples=15).fit(data)
    cluster_labels = np.array(clustering.labels_)
    # x_mode, y_mode, z_mode = mode(data, axis=0,keepdims=True).mode[0]
    # print(mode(data, axis=0,keepdims=True).mode[0])
    # data = mode(data, axis=0,keepdims=True).mode[0]
    current_centroids = calculate_centroids(data, cluster_labels)

    # Get the closest centroids and assign consistent IDs, keeping only valid ones
    valid_centroids, next_id = find_closest_centroid(previous_centroids, current_centroids, id_mapping, next_id, frame_counts, frame_threshold=10)
    previous_centroids = current_centroids  # Update previous centroids for next frame
    for centroid_id, centroid in valid_centroids:
        color = cmap(centroid_id % 20)  # Assign unique color based on ID
        ax1.scatter(centroid[0], centroid[1], centroid[2], marker='o', s=200, color=color, label=f'Cluster ID {centroid_id}')

    # x, y, z = data
    # ax1.scatter(x, y, z, color='red', marker='o', s=200)


    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plotPath}/test_{str(index)}.png")
    plt.close()
    