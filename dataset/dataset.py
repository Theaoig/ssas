import os,natsort,random,cv2,torch,json
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

class CoCoPseudoMask:
    """one can input actual img path and it's mask path, or input img path and it's Json annotation path
    """
    def __init__(self,pseudo_img='dataset/SHTech/element',pseudo_mask='dataset/SHTech/mask',random_agument=False):
        super(CoCoPseudoMask, self).__init__()
        self.pseudo_img=pseudo_img
        self.pseudo_mask=pseudo_mask
        self.random_agument=random_agument
        
    def __call__(self,origin_img):
        insert_times=random.randint(0,2)
        origin_mask=np.zeros_like(origin_img)
        if insert_times == 0:
            return origin_img,origin_mask,torch.tensor([0]).float()
        while insert_times>0:
            try:
                box_img,box_mask=self.get_random_box_and_mask()
                a=random.randint(int(0.2*origin_img.shape[0]),int(0.4*origin_img.shape[0]))
                b=random.randint(int(0.1*origin_img.shape[1]),int(0.15*origin_img.shape[1]))
                box_img=cv2.resize(box_img,(a,b))
                box_mask=cv2.resize(box_mask,(a,b))
                origin_img,origin_mask=self.insert_box_and_mask(origin_img,origin_mask,box_img,box_mask)
                insert_times-=1
            except:
                pass
        
        return origin_img,origin_mask,torch.tensor([1]).float()
    
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
        if self.pseudo_mask.endswith(".json"):
            with open(self.pseudo_mask,'r') as load_f:
                load_dict = json.load(load_f)
            while True:
                index=random.randint(0,len(load_dict['annotations'])-1)
                img_dir=os.path.join(self.pseudo_img,"{}.jpg".format(str(load_dict['annotations'][index]["image_id"]).zfill(12)))
                category_id=load_dict['annotations'][index]['category_id']
                if (os.path.exists(img_dir)) and category_id!=1:
                    try:
                        bbox=load_dict['annotations'][index]['bbox']
                        bbox=[int(x) for x in bbox]
                        bound=load_dict['annotations'][index]['segmentation'][0]
                        bound=[int(bound[i]-bbox[i%2]) for i in range(len(bound))]
                        bound=np.array(bound).reshape((-1,1,2,))
                        box_img=cv2.imread(img_dir)[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
                        box_mask=np.zeros_like(box_img)
                        cv2.polylines(box_mask, np.int32([bound]), 1, 1)
                        cv2.fillPoly(box_mask, np.int32([bound]), (255,255,255))
                        return box_img,box_mask
                    except:
                        pass
                else:
                    pass
        else:
            elements=os.listdir(self.pseudo_img)
            K=random.randint(0,len(elements)-1)
            box_img=cv2.imread(os.path.join(self.pseudo_img,elements[K]))
            if self.random_agument:
                box_img=random_img_augment(box_img)
            box_mask=cv2.imread(os.path.join(self.pseudo_mask,elements[K][0:-4]+"_mask.jpg"))
        return box_img,box_mask
        
class PMASDataset(Dataset):
    def __init__(self,dataset_path='/data/VAD/SHTech',imsize=256, sparse=1,
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.dataset_path = dataset_path
        self.imsize=imsize
        self.sparse=sparse
        self.transform_img=T.Compose([T.Resize((imsize,imsize), Image.ANTIALIAS),T.ToTensor(),
                                      T.Normalize(mean=mean,std=std)])
        self.transform_mask=T.Compose([T.Resize((imsize,imsize), Image.ANTIALIAS),T.ToTensor()])
        self.generate_presudo_mask=CoCoPseudoMask("/data/coco/images/val","/data/coco/annotations/instances_val2017.json")
        self.all_imgs = self.load_dataset_folder()

    def __getitem__(self, idx):
        img = cv2.imread(self.all_imgs[idx])
        img,mask,label = self.generate_presudo_mask(img)
        img = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0, sigmaY=0)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).convert('RGB')
        img = self.transform_img(img)
        mask = mask[:,:,0]
        mask = self.transform_mask(Image.fromarray(mask))
        return img,mask,label

    def __len__(self):
        return len(self.all_imgs)

    def load_dataset_folder(self):
        imgs = []
        folder_dir = os.path.join(self.dataset_path, "training","frames")
        for folder_name in natsort.natsorted(os.listdir(folder_dir)):
            img_dir=os.path.join(folder_dir,folder_name)
            img_fpath_list = natsort.natsorted([os.path.join(img_dir, f)
                                    for f in natsort.natsorted(os.listdir(img_dir))
                                    if f.endswith(('.jpg','.png')) 
                                    and int(f.split('.')[0])%self.sparse==0 ])
            imgs.extend(img_fpath_list)
        return list(imgs)