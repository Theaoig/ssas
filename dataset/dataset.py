import os,natsort,random,cv2,torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

def random_img_augment(img):
    size=img.shape
    a=int(2*random.randint(0,3)+1)
    b=random.uniform(11,21)
    if a>2:
        img = cv2.GaussianBlur(img,(a,a),b)
    alpha=random.uniform(0.7,1.2)
    beta=random.uniform(-20,25)
    gama=np.uint8(np.clip((alpha * img + beta), 0, 255))
    noise=np.random.randint(-1,2,size=[size[0],size[1],3])
    new_img=gama+noise
    return new_img

def insert_obj_and_mask(origin_img,origin_mask,box_img,box_mask):
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

def generate_presudo_mask(origin_img,element_path,mask_path):
    elements=os.listdir(element_path)
    insert_times=random.randint(0,2)
    origin_mask=np.zeros_like(origin_img)
    if insert_times == 0:
        return origin_img,origin_mask,torch.tensor([0]).float()
    for i in range(insert_times):
        K=random.randint(0,len(elements)-1)
        box_img=cv2.imread(os.path.join(element_path,elements[K]))
        box_mask=cv2.imread(os.path.join(mask_path,elements[K][0:-4]+"_mask.jpg"))
        a=random.randint(int(0.2*origin_img.shape[0]),int(0.4*origin_img.shape[0]))
        b=random.randint(int(0.1*origin_img.shape[1]),int(0.15*origin_img.shape[1]))
        box_img=cv2.resize(box_img,(a,b))
        box_mask=cv2.resize(box_mask,(a,b))
        origin_img,origin_mask=insert_obj_and_mask(origin_img,origin_mask,box_img,box_mask)
    
    return origin_img,origin_mask,torch.tensor([1]).float()

class PMASDataset(Dataset):
    def __init__(self,dataset_path='./SHTech',imsize=256, sparse=1,
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.dataset_path = dataset_path
        self.imsize=imsize
        self.sparse=sparse
        self.transform_img=T.Compose([T.Resize((imsize,imsize), Image.ANTIALIAS),T.ToTensor(),
                                      T.Normalize(mean=mean,std=std)])
        self.transform_mask=T.Compose([T.Resize((imsize,imsize), Image.ANTIALIAS),T.ToTensor()])
        
        self.all_imgs = self.load_dataset_folder()

    def __getitem__(self, idx):
        img = cv2.imread(self.all_imgs[idx])
        img,mask,label = generate_presudo_mask(img,'dataset/SHTech/element','dataset/SHTech/mask')
        img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).convert('RGB')
        img = self.transform_img(img)
        mask=mask[:,:,0]
        mask=self.transform_mask(Image.fromarray(mask))
        return img,mask,label

    def __len__(self):
        return len(self.all_imgs)

    def load_dataset_folder(self):
        imgs = []
        folder_dir = os.path.join(self.dataset_path, "train")
        for folder_name in natsort.natsorted(os.listdir(folder_dir)):
            img_dir=os.path.join(folder_dir,folder_name)
            img_fpath_list = natsort.natsorted([os.path.join(img_dir, f)
                                    for f in natsort.natsorted(os.listdir(img_dir))
                                    if f.endswith(('.jpg','.png')) 
                                    and int(f.split('.')[0])%self.sparse==0 ])
            imgs.extend(img_fpath_list)
        return list(imgs)