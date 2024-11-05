from PIL import Image
import glob
folder_path = "./visualization/drone_data_2/*.png"
image_files = sorted(glob.glob(folder_path))
frames = [Image.open(img) for img in image_files]
frames[0].save(
    "output10_10m.gif",
    save_all=True,
    append_images=frames[1:],
    duration=10,  # Duration between frames in milliseconds
    loop=0  # Loop 0 for infinite loop
)

print("GIF created successfully!")
