# Face_Detection
## Summary 
Implementation of paper - [YOLO5Face: Why Reinventing a Face Detector](https://arxiv.org/abs/2105.12931)
## Prepare dataset
Download [dataset](https://www.kaggle.com/datasets/mksaad/wider-face-a-face-detection-benchmark), [annotations](https://drive.google.com/file/d/1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8/view)
## Training
python3 train.py --data data/widerface.yaml --cfg models/yolov5s.yaml --weights 'pretrained models'
## Testing
### Image 
python3 detect_face.py --weights 'pretrained models' --source 'image path'
### Webcam 
python3 detect_face.py --weights 'pretrained models' --view-img
## References
https://github.com/deepcam-cn/yolov5-face
