import os,natsort,random,cv2,torch,json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, dataset
from torchvision import transforms as T


class CoCoPseudoMask:
    """one can input actual img path and it's mask path, or input img path and it's Json annotation path
    """
    def __init__(self,img_path="coco/000000007816.jpg",mask_path="mask/000000007816_mask.jpg"):
        super(CoCoPseudoMask, self).__init__()
        self.img_path=img_path
        self.mask_path=mask_path
        
    def __call__(self,origin_img):
        origin_mask=np.zeros_like(origin_img)
        box_img,box_mask=self.get_random_box_and_mask()
        a=2*random.randint(int(0.2*origin_img.shape[0]),int(0.4*origin_img.shape[0]))
        b=2*random.randint(int(0.1*origin_img.shape[1]),int(0.15*origin_img.shape[1]))
        box_img=cv2.resize(box_img,(a,b))
        box_mask=cv2.resize(box_mask,(a,b))
        origin_img,origin_mask=self.insert_box_and_mask(origin_img,origin_mask,box_img,box_mask)
    
        return origin_img,origin_mask
    
    def insert_box_and_mask(self,origin_img,origin_mask,box_img,box_mask):
        box_mask = cv2.cvtColor(box_mask, cv2.COLOR_BGR2GRAY)
        _, box_mask = cv2.threshold(box_mask, 175, 255, cv2.THRESH_BINARY)
        mask_area = np.argwhere(box_mask > 250).tolist()
        margin=[origin_img.shape[0]-box_mask.shape[0],origin_img.shape[1]-box_mask.shape[1]]

        assert margin[0]>0 and margin[1]>0,"img size must large than box"
        insert_position=[random.randint(0,margin[0]),random.randint(0,margin[1])]
        for point in mask_area:
            try:
                origin_img[insert_position[0]+point[0],insert_position[1]+point[1]]=box_img[point[0],point[1]]
                origin_mask[insert_position[0]+point[0],insert_position[1]+point[1]]=255
            except:
                pass
        return origin_img,origin_mask

    def get_random_box_and_mask(self):
        box_img=cv2.imread(self.img_path)
        box_mask=cv2.imread(self.mask_path)
        return box_img,box_mask
    

if __name__ == '__main__':
    coco=CoCoPseudoMask()
    origin_img=cv2.imread("origin_img.jpg")
    img,mask=coco(origin_img)
    cv2.imwrite("img.jpg",img)
    cv2.imwrite("mask.jpg",mask)

        
