# 使用KITTI数据训练  (runs/train_5_KITTI)
*** 训练之前记得修改train.py中--project的输出保存路径为/train_5_KITTI
*** 超参 lr0=0.001 momentum=0.9 IoU=0.5 opt=Adam

# 指定输入大小为352*352，运行结果文件夹名称对应修改
python train.py --weights weights/yolov7-tiny.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 16 --img 352 352 --cfg cfg_3/training/yolov7-tiny_GhostRCCA.yaml --name yolov7-tiny_KITTI_352_ --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam

断点继续训练
python train.py --weights weights/last.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 16 --img 384 384 --cfg cfg_3/training/yolov7-tiny_GhostRCCA.yaml --name yolov7-tiny_KITTI_640_ --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam

******************************************正片**
# baseline实验
python train.py --weights weights/yolov7-tiny.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 8 --img 640 640 --cfg cfg_4/training/yolov7-tiny.yaml --name yolov7-tiny_KITTI_ --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam

# BiFPN实验
python train.py --weights weights/yolov7-tiny.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 8 --img 640 640 --cfg cfg_4/training/yolov7-tiny_BiFPN.yaml --name yolov7-tiny_KITTI_BiFPN_ --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam
yaml中不能含中文

# WIOU实验
python train.py --weights weights/yolov7-tiny.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 8 --img 640 640 --cfg cfg_4/training/yolov7-tiny.yaml --name yolov7-tiny_KITTI_WIOU_ --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam

# RCCA实验
python train.py --weights weights/yolov7-tiny.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 8 --img 640 640 --cfg cfg_4/training/yolov7-tiny_RCCA.yaml --name yolov7-tiny_KITTI_RCCA_ --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam


# BiFPN+WIOU实验
python train.py --weights weights/yolov7-tiny.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 8 --img 640 640 --cfg cfg_4/training/yolov7-tiny_BiFPN.yaml --name yolov7-tiny_KITTI_BiFPN_WIOU_ --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam


# BiFPN+WIOU+RCCA实验
python train.py --weights weights/yolov7-tiny.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 8 --img 640 640 --cfg cfg_4/training/yolov7-tiny_RCCA_BiFPN.yaml --name yolov7-tiny_KITTI_BiFPN_WIOU_RCCA --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam


# backbone实验
python train.py --weights weights/yolov7-tiny.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 8 --img 640 640 --cfg cfg_4/training/yolov7-tiny_backbone.yaml --name yolov7-tiny_backbone --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam


# neck实验
python train.py --weights weights/yolov7-tiny.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 8 --img 640 640 --cfg cfg_4/training/yolov7-tiny_neck.yaml --name yolov7-tiny_neck --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam


# backbone+neck实验
python train.py --weights weights/yolov7-tiny.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 8 --img 640 640 --cfg cfg_4/training/yolov7-tiny_backbone_neck.yaml --name yolov7-tiny_backbone_neck --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam

# light+精度改进实验 WIOU RCCA BiFPN
python train.py --weights weights/yolov7-tiny.pt --data mydatas/KITTI_8_2/dataset.yaml --epochs 100 --batch-size 32 --img 640 640 --cfg cfg_4/training/yolov7-tiny_light_PR.yaml --name yolov7-tiny_light_PR --hyp data/hyp.scratch.tiny.yaml --workers 0 --device 0 --adam

