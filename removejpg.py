import os
from tqdm import tqdm

path = 'C:/Users/.../ImageNet-13 Original Image Dateset (1)/image/n02123159/'

files = os.listdir(path)

for file in tqdm(files):
    try:
        filename, filetype = file.split('.')
        if filetype == '.JPG':
        #if filetype == '.json':
            continue

        imgfile = os.path.join(path, file)
        pngfile = os.path.join(path, filename + '.png')
        oldjsonfile = os.path.join(path, filename + '..json')
        jsonfile = os.path.join(path, filename + '.json')
        if os.path.exists(pngfile):
        #if os.path.exists(oldjsonfile):
            print("yes")
            # print('{} deleted.'.format(imgfile))
            os.remove(imgfile)

            #print(imgfile)
            #os.rename(imgfile, jsonfile)

    except :
        print(f"does not exist.")

