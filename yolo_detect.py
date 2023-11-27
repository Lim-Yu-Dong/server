import torch
import numpy as np
import yaml

class ImageDetect:
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s', force_reload=True)
    data = {}
    #  data 를 member variable, field라고 한다...
    # 클래스 생성 시 자동으로 호출되는 생성자 함수
    def __init__(self):
        with open('coco.yaml', 'r', encoding='UTF-8') as f:
            self.data = yaml.full_load(f)['names']

    
    # self : 1. 클래스내에 정의가 되어있음, 2. 클래스이름을 가리키는 키워드
    def detect_img(self, img):
        result = self.model(img).xyxyn[0].numpy()
        return result