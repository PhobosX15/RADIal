import os
from DBReader.DBReader import SyncReader
from SignalProcessing.rpl import RadarSignalProcessing
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, freeze_support
import numpy as np

def process_folder(file_path, folder):
    print(f"Staring for folder {folder}")
    """ os.makedirs(os.path.join('output\\images', folder), exist_ok=True)
    os.makedirs(os.path.join('output\\ra', folder), exist_ok=True) """
    os.makedirs(os.path.join('output\\ra_matrix', folder), exist_ok=True)
    try:
        db = SyncReader(os.path.join(file_path, folder), tolerance=40000, silent=True)
   
        for i in range(len(db)):
            data = db.GetSensorData(i)
            RSP = RadarSignalProcessing('SignalProcessing\\CalibrationTable.npy',
                                    method='RA',device='cuda', lib="PyTorch")
            
            ra=RSP.run(data['radar_ch0']['data'],data['radar_ch1']['data'],
                    data['radar_ch2']['data'],data['radar_ch3']['data'])
            
            ra = cv2.rotate(ra,cv2.ROTATE_180 )
            ra= ra[:,125:(ra.shape[1]-125)]
            
            """ plt.imsave(os.path.join('output\\ra', folder, "RADAR"+ str(i)+'.png'), ra)
            plt.imsave(os.path.join('output\\images', folder, "RGB"+ str(i)+'.png'),data['camera']['data']) """
            np.save(os.path.join("output\\ra_matrix",folder,"RADAR"+str(i)+".npy"), ra)
    except:
        print(f"PROBLEM: Folder {folder}")

if __name__ == '__main__':
    freeze_support()
    file_path = "E:\\RADIal" ## Replace with your path
    folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]
    folders_to_process = [(file_path, folder) for folder in folders]
    with Pool(processes = 12) as pool:
        pool.starmap(process_folder, folders_to_process)

    pool.close()
    pool.join()

