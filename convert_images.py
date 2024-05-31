import cv2
import os
import ast
import multiprocessing
from multiprocessing import Pool, freeze_support
# from output folder iterate through all the folders

output = "output\\images"
folders = [f for f in os.listdir(output) if os.path.isdir(os.path.join(output, f))]

def resize_image(new_shape, folder, resized_folder):
    
    
    os.makedirs(os.path.join(resized_folder, folder), exist_ok=True)
    for file in os.listdir(os.path.join(output, folder)):
        if file.endswith(".png"):
            # Open the image file
            try:
                img = cv2.imread(os.path.join(output, folder, file))
                img = cv2.resize(img, new_shape)
                cv2.imwrite(os.path.join(resized_folder, folder, file), img)

            except Exception as e:
                print(f"Error: {e} in {file}")
                file_num = file[-5]
                RADAR_NAME = "RADAR" + file_num+".png"
                print(f"Folder: {folder}")
                # delete the RADAR NAME in output\\ra\\folder
                try:
                    os.remove(os.path.join("output\\ra", folder, RADAR_NAME))
                except FileNotFoundError:
                    print(f"File not found: {RADAR_NAME}")

if __name__ == '__main__':
    freeze_support()
    resized_folder = "resized"
    os.makedirs(resized_folder, exist_ok=True)
    os.makedirs(os.path.join(resized_folder, "images"), exist_ok=True)
    new_shape = ast.literal_eval(input("Enter the new shape of the image (width, height): "))

    folder_to_process = [(new_shape,folder, resized_folder) for folder in folders]
    with Pool(processes= 2) as pool:
        pool.starmap(resize_image, folder_to_process)

    pool.close()
    pool.join()