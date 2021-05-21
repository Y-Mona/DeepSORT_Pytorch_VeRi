# DeepSORT_Pytorch_VeRi
using pythorch , deepsort and veri datasets to train your own tracking model
## datasets
(remember to change the path of your datasets and other setting in the .py)
1. cd deepsort/deep/data
2. put your VeRi in it
3. cd ../
4. use getVal.py to get val imgs
5. cd data
6. use move2folder.py to get the satisfactory structure(train & val) 
7. this movement may have some problems,I did not modify it.you only need to move it manually to val or train.
8. the structure: 
data

——train

————0001

——————0001xxx.jpg

——val

————0001

——————0001xxx.jpg


## my modification 
https://github.com/Y-Mona/DeepSORT_Pytorch_VeRi/blob/cca50f117753b457ae9d414cc78141f7064f79cb/deep_sort/deep/train.py#L30
change the test datasets

https://github.com/Y-Mona/DeepSORT_Pytorch_VeRi/blob/cca50f117753b457ae9d414cc78141f7064f79cb/deep_sort/deep/train.py#L32
change the randomcrop to resize

## train
python train.py

## track
(require .cfg & .weight which is trained by yolov3. yolov3-spp.cfg could not run. For details,see my repo https://github.com/Y-Mona/YOLOv3_Pytorch_UA-DETRAC)

1. cd detector/YOLOv3
2. put in your cfg & weight
3. modify the input of detector.py (cfg,weight,names)
4. modify the parameters of yolov3_deepsort.py(.yaml)
5. modify the yaml you use
6. modify 'mask' of yolov3_deepsort.py(0 start,the line of the class you car tracking in xx.names)
7. python yolov3_deepsort.py xxx.avi

## result

![train(1)](https://user-images.githubusercontent.com/55938144/119069392-a5e28780-ba18-11eb-91df-dab55c6adf76.jpg)

## reference
https://github.com/ZQPei/deep_sort_pytorch
https://blog.csdn.net/weixin_40194996/article/details/104779138
