import os
import json
os.getcwd()
# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

# download cascade mask rcnn model&config
model_path = 'work_dirs/cascade_convnext_s/epoch_48.pth'
config_path = 'configs/cascade_rcnn_r50_fpn_1x.py'

# ## 4. Batch Prediction
model_type = "mmdet"
model_path = model_path
model_config_path = config_path
model_device = "cuda:0" # or 'cuda:0'
model_confidence_threshold = 0.001

slice_height = 768
slice_width = 960
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "data/test_images"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='mmdet',
    model_path=model_path,
    config_path=config_path,
    confidence_threshold=0.001,
    # image_size=(2880,2048),
    device="cuda:0", # or 'cuda:0'
)
jdict = []

fp = open('data/test.txt','r')
test_list = fp.readlines()
imgs = []

for test_1 in test_list:
    test_1 = test_1.replace('\n','')
    name = test_1
    imgs.append(name)
count = 0

for img in imgs:
    count += 1
    print('model is processing the {}/{} images.'.format(count,len(imgs)))
    result = get_sliced_prediction(
        img,
        detection_model,
        slice_height = slice_height,
        slice_width = slice_width,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2
    )

    rawjson = result.to_coco_predictions(image_id=1)
    for item in rawjson:
        bbox = item['bbox']
        float_bbox = []
        for i in bbox:
            float_bbox.append(float(i)+0.00001)
        jdict.append({'image_id':img[29:-4],
                      'category_id':item['category_id'],
                      'bbox':float_bbox,
                      'score':round(item['score'],5)})

pred_json = './pred.json'
print('\nSaveing %s...' % pred_json)
with open(pred_json, 'w') as f:
    json.dump(jdict, f)
