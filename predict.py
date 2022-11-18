import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import time
import cv2
import numpy as np
cur_time = time.time()
from model import MobileNetV2



print(torch.__version__)
data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# load image
cap=cv2.VideoCapture(0)
while(1):
    ret,frame =cap.read()
    if ret==True:
        # print (type(frame))
        print (ret)#观察frame和ret的类型
        img = Image.fromarray(frame)#完成np.array向PIL.Image格式的转换
        id=0
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        # print("img:", img)
        # read class_indict
        try:
            json_file = open('./class_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)
        # create model
        model = MobileNetV2(num_classes=7)
        # load model weights
        model_weight_path = "bestmodel.pth"
        model.load_state_dict(torch.load(model_weight_path))
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img))
            out=np.array(output)
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        print(predict_cla)
        predict_cla = int(predict_cla)
        type=['unknow','D1battery','A2injector','B1vegetable','C1mask','C2swab','A1bottle','D2cigarette']
        predict_cla=type[predict_cla]
        print(predict_cla)
        font = cv2.FONT_HERSHEY_SIMPLEX
        show=cv2.putText(frame,str(predict_cla), (10,50), font, 2, (255,255,0), 5)
        cv2.imshow('123',show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
