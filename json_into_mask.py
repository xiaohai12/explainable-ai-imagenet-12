"""
import argparse
import json
import os
import os.path as osp
import warnings
import copy
import numpy as np
import PIL.Image
from skimage import io
import yaml
from labelme import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('n2974003')   
    #parser.add_argument('n2977058_json1')  
    #parser.add_argument('n2992211_json1')   
    #parser.add_argument('n7614500_json1')  
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file

    list = os.listdir(json_file)   
    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])  
        filename = list[i][:-5]      
        extension = list[i][-4:]
        if extension == 'json':
            if os.path.isfile(path):
                data = json.load(open(path))
                img = utils.image.img_b64_to_arr(data['imageData'])  
                lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])  
                #captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
                #lbl_viz = utils.draw.draw_label(lbl, img, captions)
                out_dir = osp.basename(list[i])[:-5]+'_json'
                out_dir = osp.join(osp.dirname(list[i]), out_dir)
                if not osp.exists(out_dir):
                    os.mkdir(out_dir)
                PIL.Image.fromarray(img).save(osp.join(out_dir, '{}_source.png'.format(filename)))
                PIL.Image.fromarray(lbl).save(osp.join(out_dir, '{}_mask.png'.format(filename)))
                #PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.jpg'.format(filename)))

                with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                    for lbl_name in lbl_names:
                        f.write(lbl_name + '\n')

                warnings.warn('info.yaml is being replaced by label_names.txt')
                info = dict(label_names=lbl_names)
                with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                    yaml.safe_dump(info, f, default_flow_style=False)

                print('Saved to: %s' % out_dir)

"""
"""


import labelme

import os
import glob


def labelme2images(input_dir, output_dir, force=False, save_img=False, new_size=False):
    
    
    
    if save_img:
        _makedirs(path=osp.join(output_dir, "images"), force=force)
        if new_size:
            new_size_width, new_size_height = new_size

    print("Generating dataset")

    filenames = glob.glob(osp.join(input_dir, "*.json"))

    for filename in filenames:
        # base name
        base = osp.splitext(osp.basename(filename))[0]

        label_file = labelme.LabelFile(filename=filename)

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        h, w = img.shape[0], img.shape[1]

        if save_img:
            if new_size:
                img_pil = Image.fromarray(img).resize((new_size_height, new_size_width))
            else:
                img_pil = Image.fromarray(img)

            img_pil.save(osp.join(output_dir, "images", base + ".jpg"))



if __name__ == '__main__':
    main()

"""


import json
from labelme.utils.shape import labelme_shapes_to_label
import numpy as np
import cv2
import os

def test():
    image_origin_path = r"./home/mars/chongyu_project/Inside-Outside-Guidance-master/test_img/n123.png/" #the original pic
    image = cv2.imread(image_origin_path)

    json_path = r'./home/mars/chongyu_project/Inside-Outside-Guidance-master/test_img/n123.json/' #the json file
    data = json.load(open(json_path))

    lbl, lbl_names = labelme_shapes_to_label(image.shape, data['shapes'])
    print(lbl_names)
    mask=[]
    class_id=[]
    for i in range(1,len(lbl_names)):
        key = [k for k, v in lbl_names.items() if v == i][0]
        print(key)
        mask.append((lbl==i).astype(np.uint8))
        class_id.append(i)
    print(class_id)
    # print(mask)
    # print(class_id)
    mask=np.asarray(mask,np.uint8)
    mask=np.transpose(np.asarray(mask,np.uint8),[1,2,0])


def get_finished_json(root_dir):
    import glob
    json_filter_path = root_dir + "\*.json"
    jsons_files = glob.glob(json_filter_path)
    return jsons_files


def get_dict(json_list):
    dict_all = {}
    for json_path in json_list:
        dir,file = os.path.split(json_path)
        file_name = file.split('.')[0]
        image_path = os.path.join(dir,file_name+'.JPEG')
        dict_all[image_path] = json_path
    return dict_all


def process(dict_):
    for image_path in dict_:
        mask = []
        class_id = []
        key_ = []
        image = cv2.imread(image_path)
        json_path = dict_[image_path]
        data = json.load(open(json_path))
        lbl, lbl_names = labelme_shapes_to_label(image.shape, data['shapes'])
        for i in range(1, len(lbl_names)):
            key = [k for k, v in lbl_names.items() if v == i][0]
            mask.append((lbl == i).astype(np.uint8))
            class_id.append(i)
            key_.append(key)
        mask = np.asarray(mask, np.uint8)
        mask = np.transpose(np.asarray(mask, np.uint8), [1, 2, 0])
        image_name = os.path.basename(image_path).split('.')[0]
        dir_ = os.path.dirname(image_path)
        for i in range(0, len(class_id)):
            image_name_ = "{}.png".format(image_name,key_[i],i)
            dir_path =  os.path.join(dir_, 'mask',key_[i])
            checkpath(dir_path)
            image_path_ = os.path.join(dir_path,image_name_)
            print(image_path_)
            retval, im_at_fixed = cv2.threshold(mask[:,:,i], 0, 255, cv2.THRESH_BINARY)
            cv2.imwrite(image_path_, im_at_fixed)


def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    root_dir = r'C:/Users/qiang.i.li/Downloads/image/n02992211/' # the root
    json_file = get_finished_json(root_dir)
    image_json = get_dict(json_file)
    process(image_json)

