from PIL import Image
import glob

# File paths to the images (assuming the images are named as cluster_2.png, cluster_3.png, ..., cluster_19.png)
image_folder = 'plots/NBC/'  # Replace with your folder path
image_files = [f"{image_folder}NBC_Zelnik1_k{k}.png" for k in range(2, 20)]

# Load images
images = [Image.open(img) for img in image_files]

# Save as GIF
output_gif_path = 'clusters3.gif'  # Output GIF file name
images[0].save(
    output_gif_path,
    save_all=True,
    append_images=images[1:], 
    duration=500,  # Duration in milliseconds for each frame
    loop=0  # Infinite loop
)

print(f"GIF saved as {output_gif_path}")
