import os
import json
import pandas as pd
import numpy as np

txt_path = './conntest'
Width = 2880
Height = 2048
sub_img_row = 2
sub_img_col = 2
overlap = 0.3
images = os.listdir('./test_images')
jdict = []
for img_path in images:
    path = os.path.join(txt_path, img_path.split('.')[0] + '.txt')
    try:
        txt = pd.read_table(path, header=None, sep=' ')
        txt.loc[:, [1, 2]], txt.loc[:, [3, 4]] = (txt.loc[:, [1, 2]].values) * np.array([Width,Height]), (txt.loc[:, [3,4]].values) * np.array([Width, Height])
        image_id = img_path.split('.')[0]
        ids = txt[0].tolist()
        box = np.array(txt.loc[:,1:4])
        box[:,:2] -= box[:, 2:] / 2
        # box = np.where(box<0, 0, box)
        scores = np.array(txt.loc[:,5])
        for p, b, c in zip(scores.tolist(), box.tolist(), ids):
            jdict.append({'image_id': image_id,
                          'category_id': round(c),
                          'bbox': [round(x, 3) for x in b],
                          'score': round(p, 5)})
    except:
        pass

pred_json = '/home/pacaep/Downloads/pred.json'
print('\nSaving %s...' % pred_json)
with open(pred_json, 'w') as f:
    json.dump(jdict, f)


