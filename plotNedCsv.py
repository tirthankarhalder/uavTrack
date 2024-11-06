import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('./telemetry_data/2024-11-05_17_29_23_telemtry.csv')
print(df.shape)
for index, row in df.iterrows():
    print(f"Index: {index}")
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x_m'][index], df['y_m'][index], df['z_m'][index], c='b', marker='o')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(-3, 10)
    plt.title('3D Scatter Plot of x_m, y_m, z_m')
    plt.savefig(f"./ned/test_{str(index)}.png")
    plt.close()
