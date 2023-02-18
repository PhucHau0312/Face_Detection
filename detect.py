import argparse
from pathlib import Path
import sys
import os

import numpy as np
import cv2
import torch
import copy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, scale_coords, increment_path


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
   
    if ratio_pad is None:  
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]) 
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0] 
    coords[:, [1, 3, 5, 7, 9]] -= pad[1] 
    coords[:, :10] /= gain
    
    coords[:, 0].clamp_(0, img0_shape[1])  
    coords[:, 1].clamp_(0, img0_shape[0])  
    coords[:, 2].clamp_(0, img0_shape[1]) 
    coords[:, 3].clamp_(0, img0_shape[0])  
    coords[:, 4].clamp_(0, img0_shape[1]) 
    coords[:, 5].clamp_(0, img0_shape[0])  
    coords[:, 6].clamp_(0, img0_shape[1])  
    coords[:, 7].clamp_(0, img0_shape[0])  
    coords[:, 8].clamp_(0, img0_shape[1]) 
    coords[:, 9].clamp_(0, img0_shape[0])  
    return coords

def show_results(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect(model, device):
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            img0 = copy.deepcopy(frame)
            h0, w0 = img0.shape[:2]  
            r = img_size / max(h0, w0)
            if r != 1:  
                interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
                img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

            img = img0.transpose(2, 0, 1).copy()
            img = torch.from_numpy(img).to(device)
            img = img.float() 
            img /= 255.0 
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img)[0]
        
            pred = non_max_suppression_face(pred, conf_thres, iou_thres)
            print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')
            print(pred)
            for det in pred: 
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                    det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], frame.shape).round()

                    for j in range(det.size()[0]):
                        xyxy = det[j, :4].view(-1).tolist()
                        conf = det[j, 4].cpu().numpy()
                        landmarks = det[j, 5:15].view(-1).tolist()
                        class_num = det[j, 15].cpu().numpy()
                        
                        show_results(frame, xyxy, conf, landmarks, class_num)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT/'pretrain/yolov5n-face.pt', help='model.pt path(s)')
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    detect(model, device)
