import os


def writeTrainPath(path):
    filelist = os.listdir(path)
    with open(r'train_list.txt', 'w', encoding='utf8') as f:
        for item in filelist:
            item = str(item)
            f.write('mydatas/KITTI_8_2/images/train/' + item + '\n')


def writeValPath(path):
    filelist = os.listdir(path)
    with open(r'val_list.txt', 'w', encoding='utf8') as f:
        for item in filelist:
            item = str(item)
            f.write('mydatas/KITTI_8_2/images/val/' + item + '\n')


train_images_path = f'images/train'
writeTrainPath(train_images_path)
val_images_path = f'images/val'
writeValPath(val_images_path)
