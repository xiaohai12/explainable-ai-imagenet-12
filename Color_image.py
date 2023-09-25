# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:30:23 2023

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:37:03 2020

@author: dan
"""

import numpy as np
import cv2
import os
import random

def Color_Img():
    flag_blur = 0
    flag_bg_bright = 0
    flag_bg_color = 1
    path1=r'F:\Explainable\image'
    for dirs1 in os.listdir(path1):
        #print(dirs1)
       # if dirs1 != 'n07614500':
           # continue
        
        for files in os.listdir( os.path.join(path1, dirs1)):
            if files == 'n02992211_15171.JPEG':
                continue
            ff = os.path.join(path1, dirs1, files)# original image
            print(ff)
            orig_img = cv2.imread(ff)
            new_img = orig_img.copy()
            
            masks = ff[:-5].replace('\\image\\','\\masks\\')+'.png'
            if not os.path.exists(masks):
                print(masks)
                continue
            v_masks = cv2.imread(masks)
            if flag_blur:
                blur_img = cv2.GaussianBlur(orig_img, (9, 9), cv2.BORDER_DEFAULT)#blur
                new_img[np.where(v_masks[:,:,2]==0)] = blur_img[np.where(v_masks[:,:,2]==0)]
                cv2.imshow('demo',new_img) 
                cv2.waitKey(0)
                cv2.imwrite(path1+'_blur.png', new_img)
            if flag_bg_bright:
                blur_img = cv2.GaussianBlur(orig_img, (9, 9), cv2.BORDER_DEFAULT)#blur
                usm = cv2.addWeighted(orig_img, 1.5, blur_img, 0.9, 1.9) #提亮、变暗
                new_img[np.where(v_masks[:,:,2]==0)] = usm[np.where(v_masks[:,:,2]==0)]
                cv2.imshow('demo',new_img) 
                cv2.waitKey(0)
                cv2.imwrite(path1+'_bright.png', new_img)
            if flag_bg_color:
                img_hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
                # H空间中，绿色比黄色的值高一点，所以给每个像素+15，黄色的树叶就会变绿
                turn_hsv = img_hsv.copy()
                hue_value = [[[0,29],[150,180]],[30,89],[90,149]]
                for i in range(3):
                    if i == 0:
                        j = random.randint(0,1)
                        [h_min,h_max] = hue_value[i][j]
                    else:
                        [h_min,h_max] = hue_value[i]
                    hues = random.randint(h_min, h_max)
                    #print(hues)
                    #print(turn_green_hsv[:, :, 0]+max_v)
                    turn_hsv[:, :, 0] = hues#(turn_hsv[:, :, 0]+hues) % 180 #[[]]
                    turn_img = cv2.cvtColor(turn_hsv, cv2.COLOR_HSV2BGR)
                    new_img[np.where(v_masks[:,:,2]==0)] = turn_img[np.where(v_masks[:,:,2]==0)]
#                    cv2.imshow('demo',new_img)
#                    cv2.waitKey(0)
                    if i == 0:
                        new_path = ff.replace('\\image\\','\\image_r\\')
                    elif i== 1:
                        new_path = ff.replace('\\image\\','\\image_g\\')
                    else:
                        new_path = ff.replace('\\image\\','\\image_b\\')
                    base_newpath = os.path.dirname(new_path)
                    if not os.path.exists(base_newpath):
                        os.makedirs(base_newpath)
                    cv2.imwrite(new_path, new_img)  
                    
                    #rainbow
                    turn_hsv = img_hsv.copy()
                    hues = random.randint(0,90)
                    turn_hsv[:, :, 0] = (turn_hsv[:, :, 0]+hues) % 180 #[[]]
                    turn_img = cv2.cvtColor(turn_hsv, cv2.COLOR_HSV2BGR)
                    new_img[np.where(v_masks[:,:,2]==0)] = turn_img[np.where(v_masks[:,:,2]==0)]
#                    cv2.imshow('demo',new_img)
#                    cv2.waitKey(0)

                    new_path = ff.replace('\\image\\','\\image_rainbow\\')
                    base_newpath = os.path.dirname(new_path)
                    if not os.path.exists(base_newpath):
                        os.makedirs(base_newpath)
                    cv2.imwrite(new_path, new_img)
                
                
if __name__ == '__main__':
    Color_Img()
    
k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()
    #break  
    
