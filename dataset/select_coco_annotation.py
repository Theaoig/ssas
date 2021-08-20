import json,cv2,random,os
import numpy as np
from collections import OrderedDict

from numpy.core.numeric import roll

def roll_dice(number=2):
    dice=random.randint(1,number)
    return dice==1

def label_select(annotation_path,image_dir,select_list,sample_ratio):
    new_annotation,counter={},{}
    new_annotation['annotations']=[]
    
    with open(annotation_path,'r') as load_f:
            load_dict = json.load(load_f)
    
    id_to_name=OrderedDict()
    for item in load_dict["categories"]:
        id_to_name[item['id']]=item['name']

    for index in range(0,len(load_dict['annotations'])-1):
        label_name=id_to_name[load_dict['annotations'][index]['category_id']]
        img_dir=os.path.join(image_dir,"{}.jpg".format(str(load_dict['annotations'][index]["image_id"]).zfill(12)))
        if label_name not in select_list or not os.path.exists(img_dir) or not roll_dice(sample_ratio[label_name]):
            continue
        if len(load_dict['annotations'][index]['segmentation'])==1 and load_dict['annotations'][index]['iscrowd']==0:
            new_annotation['annotations'].append({'bbox':load_dict['annotations'][index]['bbox'],
                                                'segmentation':load_dict['annotations'][index]['segmentation'][0],
                                                'label_name':label_name,
                                                'img_dir':img_dir})
            if label_name not in counter.keys():
                counter[label_name]=1
            else:
                counter[label_name]+=1
    new_annotation['distributions']=counter
    return new_annotation

if __name__=="__main__":
    select_names = ['person','bicycle','car','motorcycle','truck']
    sample_ratio = {'person':50,'bicycle':1,'car':9,'motorcycle':1,'truck':2}

    annotation_path = "/data/coco/annotations/instances_train2017.json"
    image_dir = "/data/coco/images/train"
    
    new_annotation=label_select(annotation_path,image_dir,select_names,sample_ratio)
    print(new_annotation['distributions'])
    with open("selected_coco_annotation.json",'w') as load_f:    
        json.dump(new_annotation,load_f)

