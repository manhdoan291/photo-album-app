import os

image_dir = "/home/doan/Pictures/id_regcognition/cccd"
image_names = os.listdir(image_dir)
for image_name in image_names:
    image_path = os.path.join(image_dir, image_name)
    print(image_path)