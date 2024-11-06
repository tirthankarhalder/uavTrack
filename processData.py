import json
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import os
from sklearn.cluster import DBSCAN
# from matplotlib.cm import cm
# from scipy.stats import mode
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta


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
if __name__ == "__main__":

    visulization = False
    previous_centroids = []
    id_mapping = {}
    frame_counts = {}
    next_id = 0
    filePath = "./data/drone_data_2.txt"
    visFile = filePath.split("/")[-1].split(".")[0]
    data = []

    with open(filePath, "r") as file:
        for line in file:
            data.append(json.loads(line)["answer"])
    radarData = pd.DataFrame(data)
    
    newDF = radarData
    radarData['datenow'] = pd.to_datetime(radarData['datenow'], format='%m/%d/%Y')
    radarData['timenow'] = pd.to_datetime(radarData['timenow'].str.replace('_', ':'), format='%H:%M:%S').dt.time
    radarData['datetime'] = radarData.apply(lambda row: datetime.combine(row['datenow'], row['timenow']), axis=1)
    radarData['datetime'] = radarData['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    radarData = radarData[["datetime","dopplerIdx","x_coord","y_coord","z_coord", "rp_y","noiserp_y","snrDB","noiseDB"]]

    start_time = radarData["datetime"][0]#add milisecond
    start_time_obj = datetime.strptime(start_time,'%Y-%m-%d %H:%M:%S')
    frameID = 0
    time_frames = []
    for index,row in radarData.iterrows():
        time_current = start_time_obj+timedelta(seconds=frameID*(100)/1000)
        datetime_obj = pd.to_datetime(time_current, format='%Y-%d-%m %H:%M:%S.%f')
        new_datetime_obj = datetime_obj + relativedelta(months=1)
        new_timestamp = new_datetime_obj.strftime('%Y-%m-%d %H:%M:%S.%f')
        time_frames.append(time_current.strftime('%Y-%d-%m %H:%M:%S.%f'))
        radarData.loc[index, 'datetime']  = time_current.strftime('%Y-%d-%m %H:%M:%S.%f') 

        frameID +=1
    print("Dropping NAN rows")
    radarData = radarData.dropna()
    for index, row in radarData.iterrows():
        if sum(radarData["dopplerIdx"][index]) == 0:
            radarData.drop(index=index)
    # radarData.to_csv("radarData.csv",index=False)
    teleData = pd.read_csv('./telemetry_data/2024-11-05_17_29_23_telemtry.csv')
    teleData = teleData[["datetime","x_m","y_m","z_m","roll_rad_s","pitch_rad_s","yaw_rad_s"]]

    radarData['datetime'] = pd.to_datetime(radarData['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    teleData['datetime'] = pd.to_datetime(teleData['datetime'])
    print(radarData['datetime'][7],teleData['datetime'][7])

    mergedTeleRadar = pd.merge_asof(radarData, teleData, on='datetime', direction='nearest')
    mergedTeleRadar =mergedTeleRadar.dropna(subset=['x_m'])
    print("teleData.shape: ", teleData.shape)
    print("radarData.shape: ", radarData.shape)
    print("200ms - mergedTeleRadar after dropna.shape: ",mergedTeleRadar.shape)
    # mergedTeleRadar.to_csv("mergedTeleRadar.csv",index=False)
    # cmap = plt.colormaps.get_cmap("tab20", 20)
    cmap = cm.get_cmap("tab20", 20)
    if visulization :
        
        for index,row in mergedTeleRadar.iterrows():
            
            frames = 10
            if index >= len(mergedTeleRadar) - frames:
                break
            x_coords = [e for elm in mergedTeleRadar['x_coord'][index:index + frames] for e in elm]
            y_coords = [e for elm in mergedTeleRadar['y_coord'][index:index + frames] for e in elm]
            z_coords = [e for elm in mergedTeleRadar['z_coord'][index:index + frames] for e in elm]
            doppler_idx = [e for elm in mergedTeleRadar['dopplerIdx'][index:index + frames] for e in elm]
            colors = [cmap(val % 20) for val in doppler_idx]
            sns.set(style="whitegrid")
            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(121,projection='3d')
            img1 = ax1.scatter(x_coords, y_coords, z_coords, c = colors, marker='o')
            fig.colorbar(img1)
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
            current_centroids = calculate_centroids(data, cluster_labels)

            # Get the closest centroids and assign consistent IDs, keeping only valid ones
            valid_centroids, next_id = find_closest_centroid(previous_centroids, current_centroids, id_mapping, next_id, frame_counts, frame_threshold=10)
            previous_centroids = current_centroids  # Update previous centroids for next frame
            for centroid_id, centroid in valid_centroids:
                color = cmap(centroid_id % 20)  # Assign unique color based on ID
                ax1.scatter(centroid[0], centroid[1], centroid[2], marker='o', s=200, color=color, label=f'Cluster ID {centroid_id}')
            print(f"Index: {index}")
            ax2 = fig.add_subplot(122,projection='3d')
            x_t = [elm for elm in mergedTeleRadar['x_m'][index:index + frames]]
            print(x_t)
            y_t = [elm for elm in mergedTeleRadar['y_m'][index:index + frames]]
            z_t = [elm for elm in mergedTeleRadar['z_m'][index:index + frames]]  
            img2 = ax2.scatter(x_t, y_t, z_t,marker='o')
            fig.colorbar(img2)
            ax2.set_title('Telemetry UAV')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_xlim(-10, 10)
            ax2.set_ylim(0, 10)
            ax2.set_zlim(-3, 10)


            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./visualization/merged/radarDepth__{str(index)}.png")
            # plt.show()
            plt.close()
            # if index ==3:
            #     break
        print("Sample Visulization Saved")
    

    # plotPath = "./visualization/" + visFile
    # os.makedirs(plotPath, exist_ok=True)

   

    # # Define a color map for unique IDs
    # cmap = cm.get_cmap("tab20", 20)


    # for index, row in df.iterrows():
    #     frames = 10
    #     # Add break condition if there aren't enough frames left to process
    #     if index >= len(df) - frames:
    #         break
    #     print('index:', index)
    #     sns.set(style="whitegrid")
    #     fig = plt.figure(figsize=(12, 6))
    #     ax1 = fig.add_subplot(111, projection='3d')
    #     x_coords = [e for elm in df['x_coord'][index:index + frames] for e in elm]
    #     y_coords = [e for elm in df['y_coord'][index:index + frames] for e in elm]
    #     z_coords = [e for elm in df['z_coord'][index:index + frames] for e in elm]
    #     doppler_idx = [e for elm in df['dopplerIdx'][index:index + frames] for e in elm]
    #     # print(doppler_idx)
    #     # break
    #     colors = [cmap(val % 20) for val in doppler_idx]
        
    #     img1 = ax1.scatter(x_coords, y_coords, z_coords, c=colors, marker='o')
    #     fig.colorbar(img1, ax=ax1, label='Intensity')
    #     ax1.set_title('Radar PCD')
    #     ax1.set_xlabel('X')
    #     ax1.set_ylabel('Y')
    #     ax1.set_zlabel('Z')
    #     ax1.set_xlim(-10, 10)
    #     ax1.set_ylim(0, 10)
    #     ax1.set_zlim(-3, 10)

    #     data = np.array([x_coords, y_coords, z_coords]).T
    #     clustering = DBSCAN(eps=1, min_samples=15).fit(data)
    #     cluster_labels = np.array(clustering.labels_)
    #     current_centroids = calculate_centroids(data, cluster_labels)

    #     # Get the closest centroids and assign consistent IDs, keeping only valid ones
    #     valid_centroids, next_id = find_closest_centroid(previous_centroids, current_centroids, id_mapping, next_id, frame_counts, frame_threshold=10)
    #     previous_centroids = current_centroids  # Update previous centroids for next frame
    #     for centroid_id, centroid in valid_centroids:
    #         color = cmap(centroid_id % 20)  # Assign unique color based on ID
    #         ax1.scatter(centroid[0], centroid[1], centroid[2], marker='o', s=200, color=color, label=f'Cluster ID {centroid_id}')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"{plotPath}/test_{str(index)}.png")
    #     plt.close()
        