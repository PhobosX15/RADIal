import os
from DBReader.DBReader import SyncReader
from SignalProcessing.rpl import RadarSignalProcessing
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, freeze_support

def process_folder(file_path, folder):
    print(f"Staring for folder {folder}")
    os.makedirs(os.path.join('output\\images', folder), exist_ok=True)
    os.makedirs(os.path.join('output\\ra', folder), exist_ok=True)

    try:
        db = SyncReader(os.path.join(file_path, folder), tolerance=40000, silent=True)
   
        for i in tqdm(range(len(db)), desc=f"Processing and Saving images {folder}", total=len(db)):
            data = db.GetSensorData(i)
            RSP = RadarSignalProcessing('SignalProcessing\\CalibrationTable.npy',
                                        method='RA',device='cpu')
            ra=RSP.run(data['radar_ch0']['data'],data['radar_ch1']['data'],
                    data['radar_ch2']['data'],data['radar_ch3']['data'])
            ra = cv2.rotate(ra,cv2.ROTATE_180 )
            plt.imsave(os.path.join('output\\ra', folder, "RADAR"+ str(i)+'.png'), ra)
            plt.imsave(os.path.join('output\\images', folder, "RADAR"+ str(i)+'.png'),data['camera']['data'])
    except:
        print(f"PROBLEM: Folder {folder}")

if __name__ == '__main__':
    freeze_support()
    file_path = "C:\\Users\\nxg05733\\RADIAL-data" ## Replace with your path
    folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]
    folders_to_process = [(file_path, folder) for folder in folders]
    print("Here")
    with Pool(processes = 2) as pool:
        print("Here2")
        pool.starmap(process_folder, folders_to_process)

    pool.close()
    pool.join()

