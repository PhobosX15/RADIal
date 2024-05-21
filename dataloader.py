# %%
import os
from DBReader.DBReader import SyncReader
from SignalProcessing.rpl import RadarSignalProcessing
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# %%
file_path = "C:\\Users\\nxg05733\\RADIAL-data" ## REPLACE with your path
# make a image folder
os.makedirs('output\\images', exist_ok=True)

# make a ra folder
os.makedirs('output\\ra', exist_ok=True)

# get all folders in the directory
folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]

# iterate over all folders
for folder in folders:

    # make a sub_folder in output for image and ra
    os.makedirs(os.path.join('output\\images', folder), exist_ok=True)
    os.makedirs(os.path.join('output\\ra', folder), exist_ok=True)

    # get the folder inside the folder
    sub_folders = [f for f in os.listdir(os.path.join(file_path, folder))
                    if os.path.isdir(os.path.join(file_path, folder, f))]
    
    for sub_folder in sub_folders:

        # get all files in the folder
        db=SyncReader(os.path.join(file_path, folder, sub_folder), tolerance=40000,silent=True)
       
        
        for i in tqdm(range(len(db)), desc="Processing and Saving images", total=len(db)):
            try:
                data = db.GetSensorData(i)
            except:
                print(f"PROBLEM: Folder {sub_folder}, index {i}")
                continue

            # obtain the RA plots with CPU (only for testing on work laptop)
            """ RSP = RadarSignalProcessing('SignalProcessing\\CalibrationTable.npy',
                                        method='RA',device='cpu',) # REPLACE with path to CalibrationTable.npy """
            # obtain the RA plots with GPU (assuming PyTorch)
            RSP = RadarSignalProcessing('SignalProcessing\\CalibrationTable.npy',
                                        method='RA',device='cuda', lib="PyTorch") # REPLACE with path to CalibrationTable.npy
            
            ra=RSP.run(data['radar_ch0']['data'],data['radar_ch1']['data'],
                    data['radar_ch2']['data'],data['radar_ch3']['data'])
            
            # Flipping the image to match the orientation of the camera
            ra = cv2.rotate(ra,cv2.ROTATE_180 )

            # save the RA plot in the ra folder under the sub_folder
            plt.imsave(os.path.join('output\\ra', sub_folder, "RADAR"+ str(i)+'.png'), ra)

            # save the images in the image folder under the sub_folder
            plt.imsave(os.path.join('output\\images', sub_folder, "RADAR"+ str(i)+'.png'),data['camera']['data'])
            
        
          

       

    


