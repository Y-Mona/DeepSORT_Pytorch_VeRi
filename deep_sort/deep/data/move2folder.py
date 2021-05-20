import os
import shutil

os.chdir("DeepSORT_Pytorch_VeRi/deep_sort/deep/data/")

for f in os.listdir("train"):
    folderName = f[0:4]

    if not os.path.exists(folderName):
        os.mkdir(folderName)
        shutil.move(os.path.join('train', f), os.path.join(folderName))
    else:
        shutil.move(os.path.join('train', f), os.path.join(folderName))
