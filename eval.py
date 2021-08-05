import os,sys,natsort,argparse,time,subprocess,cv2
import torch,torchvision,einops
import numpy as np
from tqdm import tqdm
from PIL import Image,ImageFilter
from torchvision import transforms as T
from datetime import datetime
from multiprocessing import Pool
from model.pmas import PMAS
from sklearn.metrics import roc_auc_score
from dataset.dataset import PMASDataset
from torch.utils.data import DataLoader


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

def evaluation(model,gt_path,dataset_path,batch_size,im_size,device,slide_avg=0.25):
    ground_th=np.load(gt_path)
    if not gt_path.split('.')[0].endswith('shanghai'):
        ground_th=ground_th[0]
        
    dataset = PMASDataset(dataset_path=dataset_path,imsize=im_size, sparse=1, phase="test")
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=16)
    score_list=[]
    with torch.no_grad():
        for i, (img) in tqdm(enumerate(loader),"processing evaluation",total=None,leave=False,unit="img"):
            img=img.to(args.device)
            pred_mask,pred_label=model(img)
            pred_label=pred_label[:,0]
            score_list+=pred_label.cpu().detach().numpy().tolist()

    auc=roc_auc_score(ground_th[0:len(score_list)],score_list)
    return auc
    
def main(args):
    Dict={"SHTech":"frame_labels_shanghai.npy",
          "Ped2":"frame_labels_ped2.npy",
          "Avenue":"frame_labels_avenue.npy"}
    gt_path=os.path.join(os.path.dirname(args.dataset_path),Dict[os.path.basename(args.dataset_path)])
    
    model=PMAS().to(args.device)
    model.load_state_dict(torch.load(args.weights)[0])
    model.eval()
    auc=evaluation(model,gt_path,args.dataset_path,args.batch_size,args.imsize,args.device)
    print("\n=> evaluation auc: {:.3f}.".format(auc))
    
if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/VAD/SHTech', help='dataset path')
    parser.add_argument('--batch_size', type=int, default='2')
    parser.add_argument('--weights', type=str, default='result/checkpoint/best.pt', help='model.pt path(s)')
    parser.add_argument('--imsize', type=int, default=256, help='inference size (pixels)')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()
    print(args)
    main(args)
    