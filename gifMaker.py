from PIL import Image
import glob
folder_path = "./visualization/merged/*.png"
image_files = sorted(glob.glob(folder_path))
frames = [Image.open(img) for img in image_files]
frames[0].save(
    "merger.gif",
    save_all=True,
    append_images=frames[1:],
    duration=2,  # Duration between frames in milliseconds
    loop=0  
)

print("GIF created successfully!")
