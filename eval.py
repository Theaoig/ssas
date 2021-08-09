import os,sys,natsort,argparse,time,subprocess,cv2
import torch,torchvision,einops
import numpy as np
from tqdm import tqdm
from model.pmas import PMAS
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from dataset.dataset import MVTecDataset
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

def evaluation(model,dataset_path,batch_size,im_size,device,slide_avg=0.25):
    all=['cable', 'leather', 'transistor', 'hazelnut', 'pill', 'wood',
            'toothbrush', 'bottle', 'zipper', 'carpet', 'screw', 
            'tile', 'metal_nut', 'grid', 'capsule']
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]
    AUC1,AUC2=[],[]
    for class_name in all:
        dataset = MVTecDataset(dataset_path=dataset_path,imsize=im_size, phase="test",class_name=class_name)
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=1)
        print("=> '{}' total num: {}".format(class_name,len(dataset)))
        score_list=[]
        gt_list=[]

        map_list=[]
        map_gt=[]
        with torch.no_grad():
            for i, (img,mask,label) in tqdm(enumerate(loader),"=> processing evaluation"):
                img=img.to(device)
                pred_mask,pred_label=model(img)

                score_list+=pred_label.cpu().detach().flatten().tolist()
                gt_list+=label.detach().flatten().tolist()

                map_list+=pred_mask.cpu().detach().flatten().tolist()
                map_gt+=mask.detach().flatten().tolist()
        print("=> inference done, calculating AUC...")
        fpr,tpr, _ = roc_curve(gt_list, score_list)
        img_auc = roc_auc_score(gt_list, score_list)
        AUC1.append(img_auc)
        fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, img_auc))
        fig_img_rocauc.legend(loc="lower right")
        
        fpr,tpr, _ = roc_curve(map_gt, map_list)
        pix_auc = roc_auc_score(map_gt, map_list)
        AUC2.append(pix_auc)
        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, pix_auc))
        fig_pixel_rocauc.legend(loc="lower right")
        print("=> '{}' image auc: {:.3f}, pixel auc: {:.3f}.\n".format(class_name,img_auc,pix_auc))
        
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.array(AUC1).mean())
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.array(AUC2).mean())
    fig.tight_layout()
    fig.savefig('result/eval_roc_curve.jpg', dpi=100)
    return np.array(AUC1).mean(),np.array(AUC2).mean()
    
def main(args):
    model=PMAS().to(args.device)
    model.load_state_dict(torch.load(args.weights)[0])
    model.eval()
    img_auc,pix_auc=evaluation(model,args.dataset_path,args.batch_size,args.imsize,args.device)
    print("=> average image auc: {:.3f}, average pixel auc: {:.3f}.\n".format(img_auc,pix_auc))
    
if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/VAD/mvtec', help='dataset path')
    parser.add_argument('--batch_size', type=int, default='40')
    parser.add_argument('--weights', type=str, default='result/checkpoint/best.pt', help='model.pt path(s)')
    parser.add_argument('--imsize', type=int, default=256, help='inference size (pixels)')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()
    print()
    print(args)
    main(args)
    