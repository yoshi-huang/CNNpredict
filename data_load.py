import numpy as np
import os
from tqdm import tqdm

def data_loader():

    directory = r"dataset"
    genres = ["blues", "classical", "country", "disco", "hiphop", 
          "jazz", "metal", "pop", "reggae", "rock"]
    
    x,y = [], []
    for _, _, files in os.walk(directory): continue
    for file in tqdm(files):

        path = directory+"\\"+file
        if l := np.size(np.load(path)/100) != 330752 : print(file,l)
        x.append(np.load(path)/100)
    
        genre = file.split("_")[0]
        y.append([genre == i for i in genres])

        #print(x,y)
        #time.sleep(5)

    print(len(x))
    x_data = np.array(x).astype(np.float32)
    y_data = np.array(y).astype(np.float32)
    np.save(r"data\x_data", x_data)
    np.save(r"data\y_data", y_data)
    print("data loading compelete!")

data_loader()
