# explainable-ai-imagenet-12
Enlarge the dataset to understand how image background effect the Computer Vision ML model. With the following topics: Blur Background / Segmented Background / AI generated Background/ Bias of tools during annotation/ Color in Background / Dependent Factor in Background/ LatenSpace Distance of Foreground/ Random Background with Real Environment!

## Contribution strategy (read before you contribute!!!)
1. Create a new branch with your name if it is the first time, otherwise, branch out from 'master'
2. Modify 'changelog.md' to add your changes
3. Open a Pull Request into master and would be better to ask another contributor for review. 


## File & function

1. showmask.py: show the mask of the image, you can use it for showing the segmentation mask generated with fully back mask image
2. removebg.py: remove the jpg image
3. transparent.py: remove the background of the image and convert it into transparent form and save the output as png image
4. Segmentbackground.py: generate fully transpanrent background image ( currently the one used during the demostration)
5. json_into_mask.py: convert the json file into mask image
6. segementwithRGB.py: the code NOT used for generating the background with real environment, bit-wise and operation cannot create optimal result
7. generate_background.py: generate the background with real environment, the code used for generating the background with real environment,the addWeighted method is used for generating the optimal result and it is the best method for generating the background with real environment, it is blending the image with the weight of the image, which performs an element-wise addition of the two images with equal weights.