import sys
import os
import time
from utils import *
import numpy as np

count=0
time_total=0

if __name__ == "__main__":
    file_dir = "/share/home/zhanglab/public/databases/trRosetta_dataset/a3m"#sys.argv[0]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            count+=1
            print(count,'/',len(files))
            if file.endswith(".a3m"):
                start = time.time()
                abs_path = os.path.join(file_dir,file)
                feature = preprocess(abs_path) # [1,526,140,140]
                to_dir=os.path.join(os.getcwd(),"feature")
                if not os.path.exists(to_dir):
                    os.mkdir(to_dir)
                data = np.save(os.path.join(to_dir,file[:-4]+".npy"),feature.cpu()),
                end = time.time()
                time_total+=end-start
                print("timecost:",end-start,'s')
    print("totalcpst:",time_total)