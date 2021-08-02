import os,sys,natsort,argparse,time,subprocess,cv2
import torch,torchvision,einops
import numpy as np
from PIL import Image,ImageFilter
from torchvision import transforms as T
from datetime import datetime
from multiprocessing import Pool
from model.pmas import PMAS
from sklearn.metrics import roc_auc_score

def ae_preprocess(img_path,im_size,device):
    img = cv2.imread(img_path)
    img = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0, sigmaY=0)
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).convert('RGB')
    trans=T.Compose([T.Resize((im_size,im_size), Image.ANTIALIAS),
                T.ToTensor(),T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    img=trans(img)
    img_tensor=img[np.newaxis,:]
    return img_tensor.to(device)

def normalize_psnr(psnr_list,inverse=False):
    result=[]
    Max=max(psnr_list)
    Min=min(psnr_list)
    for i in range(len(psnr_list)):
        temp=(psnr_list[i]-Min)/(Max-Min)
        if not inverse:
            result.append(temp)
        else:
            result.append(1-temp)
    return result

def evaluation(model,gt_path,dataset_path,im_size,device,slide_avg=0.25):
    ground_th=np.load(gt_path)
    if not gt_path.split('.')[0].endswith('shanghai'):
        ground_th=ground_th[0]
        
    dataset_path=os.path.join(dataset_path,"testing","frames")
    score_list=[]
    mask_score_list=[]
    for i,folder_dir in enumerate(natsort.natsorted(os.listdir(dataset_path))):
        temp1=[]
        temp2=[]
        for j,img_dir in enumerate(natsort.natsorted(os.listdir(os.path.join(dataset_path,folder_dir)))):
            sys.stdout.write("\r=> processing at %dth folder, now at: %d" %(i+1,j+1))
            sys.stdout.flush()
            x=ae_preprocess(os.path.join(dataset_path,folder_dir,img_dir),im_size,device)
            mask,label=model(x)
            score1=round(label.item()*1000, 2)
            score2=round(mask.mean().item()*1000, 2)
            try:
                temp1.append(round(slide_avg*score1+(1-slide_avg)*temp1[-1],2))
                temp2.append(round(slide_avg*score2+(1-slide_avg)*temp2[-1],2))
            except:
                temp1.append(score1)
                temp2.append(score2)
        print(" ")
        score_list+=temp1
        mask_score_list+=temp2
        if i+1==500:
            break
        
    auc=roc_auc_score(ground_th[0:len(score_list)],score_list)
    mask_auc=roc_auc_score(ground_th[0:len(mask_score_list)],mask_score_list)
    return auc,mask_auc
    
def main(args):
    Dict={"SHTech":"frame_labels_shanghai.npy",
          "Ped2":"frame_labels_ped2.npy",
          "Avenue":"frame_labels_avenue.npy"}
    gt_path=os.path.join(os.path.dirname(args.dataset_path),Dict[os.path.basename(args.dataset_path)])
    
    model=PMAS().to(args.device)
    model.load_state_dict(torch.load(args.weights)[0])
    model.eval()
    auc1,auc2=evaluation(model,gt_path,args.dataset_path,args.imsize,args.device)
    print("evaluation auc: {:.3f}. mask auc: {:.3f}".format(auc1,auc2))
    
if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/VAD/SHTech', help='dataset path')

    parser.add_argument('--weights', type=str, default='result/checkpoint/best.pt', help='model.pt path(s)')
    parser.add_argument('--imsize', type=int, default=256, help='inference size (pixels)')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()
    print(args)
    main(args)
    