import os
import random
import json
os.getcwd()
# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

# download cascade mask rcnn model&config
model_path = 'work_dirs/cascade_convnext_b_2/latest.pth'
config_path = 'configs/cascade_rcnn_r50_fpn_1x.py'

# ## 4. Batch Prediction
model_type = "mmdet"
model_path = model_path
model_config_path = config_path
model_device = "cuda:0" # or 'cuda:0'
model_confidence_threshold = 0.4

slice_height = 768
slice_width = 960
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "data/test_images/"

predict(
    model_type=model_type,
    model_path=model_path,
    model_config_path=config_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
)
