import cv2
import os
import ast
import multiprocessing
from multiprocessing import Pool, freeze_support
import numpy as np
import matplotlib.pyplot as plt
# from output folder iterate through all the folders



def resize_image( folder, resized_folder, output, type = "Image", new_shape = (224,224)):
    
    
    os.makedirs(os.path.join(resized_folder, folder), exist_ok=True)
    if type == "Image":
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
                    RADAR_NAME = "RADAR" + file_num+".npy"
                    print(f"Folder: {folder}")
                    # delete the RADAR NAME in output\\ra\\folder
                    try:
                        os.remove(os.path.join("/data/pavan/output/ra_matrix", folder, RADAR_NAME))
                    except FileNotFoundError:
                        print(f"File not found: {RADAR_NAME}")
    
    else:
        for file in os.listdir(os.path.join(output, folder)): 
            if file.endswith(".npy"):
                try:
                    ra = np.load(os.path.join(output, folder,file))
                    ra = cv2.resize(ra, new_shape)
                    ra = normalize(ra)
                    np.save(os.path.join(resized_folder,folder,file), ra)
                  
                
                except Exception as e:
                    print(f"Error: {e} in {file}")



def normalize(image):
    image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return image

if __name__ == '__main__':
    freeze_support()
    # Do Radar first
    print("Processing RADAR")
    new_shape = ast.literal_eval(input("Enter the new shape of the image (width, height): "))
    output = "/data/pavan/output/ra_matrix"
    folders = [f for f in os.listdir(output) if os.path.isdir(os.path.join(output, f))]
    resized_folder = "/data/pavan/resized_384/ra_matrix"
    os.makedirs(resized_folder, exist_ok=True)
    #os.makedirs(os.path.join(resized_folder, "images"), exist_ok=True)
    folder_to_process = [(folder, resized_folder,output,"RADAR", new_shape) for folder in folders]
    with Pool(processes= 12) as pool:
        pool.starmap(resize_image, folder_to_process)
    pool.close()
    pool.join()
    print("Done Processing RADAR")
    # Processing Image     
    print("Processing Images")   
    #new_shape = ast.literal_eval(input("Enter the new shape of the image (width, height): "))
    output = "/data/pavan/output/images"
    folders = [f for f in os.listdir(output) if os.path.isdir(os.path.join(output, f))]
    resized_folder = "data/pavan/resized_384/images"
    os.makedirs(resized_folder, exist_ok=True)
    #os.makedirs(os.path.join(resized_folder, "images"), exist_ok=True)
    folder_to_process = [(folder, resized_folder,output,"Image",new_shape) for folder in folders]
    with Pool(processes= 12) as pool:
        pool.starmap(resize_image, folder_to_process)
    pool.close()
    pool.join()
    print("Done Processing Image")

    
