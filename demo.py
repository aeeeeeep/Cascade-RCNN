import mmcv
import os
import numpy as np
import json
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
# from mmdet.apis import init_detector, inference_detector
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/cascade_rcnn_r50_fpn_1x.py'
checkpoint_file = 'work_dirs/cascade_convnext_s_2/latest.pth'

model = init_detector(config_file,checkpoint_file)
print(model.CLASSES)

img_dir = 'data/sub_images/'
out_dir = 'results/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

fp = open('data/test.txt','r')
test_list = fp.readlines()

imgs=[]
for test_1 in test_list:
    test_1 = test_1.replace('\n','')
    name = img_dir + test_1
    imgs.append(name)

jdict = []

count = 0
for img in imgs:
    count += 1
    print('model is processing the {}/{} images.'.format(count,len(imgs)))
    result = inference_detector(model,img)
    image_id = img[16:-4]

    a = np.empty(shape=(0,5))
    cls = 0
    for box in result:
        if box != a:
            for i in box:
                bbox = i[:4]
                score = i[-1:]
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                bbox = bbox.tolist()
                score = float(score[0])
                jdict.append({'image_id': image_id,
                              'category_id': cls,
                              'bbox': bbox,
                              'score': round(score,5)})
        cls += 1

pred_json = './pred.json'
print('\nSaving %s...' % pred_json)
with open(pred_json, 'w') as f:
        json.dump(jdict, f)
