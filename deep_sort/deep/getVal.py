import os
import shutil
import glob
#your absolute path
#"DeepSORT_Pytorch_VeRi/deep_sort/deep/data/val/"

import random

for xmlPath in sorted(glob.glob('...../DeepSORT_Pytorch_VeRi/deep_sort/deep/data/train' +"/*")):
    imgname = xmlPath[-4:]

    print(imgname)
    valDir=os.path.join('...../DeepSORT_Pytorch_VeRi/deep_sort_pytorch/deep_sort/deep/data/val/', imgname)
    if not os.path.exists(valDir):
        os.mkdir(valDir)
    list = sorted(glob.glob(xmlPath+"/*.jpg"))
    l=len(list)
    kk = 0
    rate=0.2

    picknumber = int (l*rate)
    sample =random.sample(list,picknumber)
    for name in sample:
        shutil.move(name,os.path.join('...../DeepSORT_Pytorch_VeRi/deep_sort/deep/data/val/',imgname))





