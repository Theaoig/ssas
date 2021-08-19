import os,sys,natsort,argparse,time,subprocess,cv2,pickle 
import torch,torchvision,einops
import numpy as np
from tqdm import tqdm
from model.pmas import PMAS
from sklearn.metrics import roc_auc_score,roc_curve
from dataset.dataset import PMASDataset
from torch.utils.data import DataLoader

def normalize(score, inverse=False):
    score=np.array(score,np.float16)
    _range=np.max(score)-np.min(score)
    score=(score-np.min(score))/(_range+1e-5) if not inverse else (np.max(score)-score)/(_range+1e-5)
    return score.tolist()

def moving_avg(score,K=0.35):
    for i in range(1,len(score)):
        score[i]=K*score[i]+(1-K)*score[i-1]
    return score

def evaluation(model,dataset_path,batch_size,im_size,device,show_log=True):
    select_path={"SHTech":"frame_labels_shanghai.npy"}
    gt_path=os.path.join("/data/VAD",select_path[os.path.basename(dataset_path)])
    
    Img_gt=np.load(gt_path)
    if not gt_path.endswith('shanghai.npy'):
        Img_gt=Img_gt[0]

    Img_score=[]
    for folder in sorted(os.listdir(os.path.join(dataset_path,"testing","frames"))):
        dataset = PMASDataset(dataset_path=dataset_path,imsize=im_size, phase="test",class_name=folder)
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=16)
        if show_log:
            print("=> '{}' total num: {}".format(folder,len(dataset)))
        
        score_list=[]
        with torch.no_grad():
            if show_log:
                for i, (img) in tqdm(enumerate(loader),"=> processing evaluation"):
                    img=img.to(device)
                    pred_mask,pred_label=model(img)
                    score_list+=pred_label.cpu().detach().flatten().tolist()
            else:
                for img in loader:
                    img=img.to(device)
                    pred_mask,pred_label=model(img)
                    score_list+=pred_label.cpu().detach().flatten().tolist()

        Img_score+=normalize(moving_avg(score_list))

    with open(r"result/eval_result.pkl", "wb") as f:
        pickle.dump([Img_gt,Img_score], f)
    auc=roc_auc_score(Img_gt,Img_score)
    return round(auc,3)
    
def main(args):
    a=time.time()
    model=PMAS().to(args.device)
    model.load_state_dict(torch.load(args.weights)[0])
    model.eval()
    img_auc=evaluation(model,args.dataset_path,args.batch_size,args.imsize,args.device)
    print("=> label auc: {:.3f}.\n".format(img_auc))
    print("total cost: {:.2f}s".format(time.time()-a))
    
def exam_mask_and_gt():
    """run this func to show that score from mask is equally with ground truth
    """
    Image_score=[]
    for folder in sorted(os.listdir(os.path.join("/data/VAD/SHTech","testing","frames"))):
        mask=np.load("/data/VAD/SHTech/testing/test_pixel_mask/{}.npy".format(folder))
        print(folder,mask.shape)
        for i in range(mask.shape[0]):
            score=1 if mask[i,:,:].sum()>0 else 0
            Image_score.append(score)    

    Image_gt=np.load("/data/VAD/frame_labels_shanghai.npy")
    print(len(Image_gt),len(Image_score))

    auc=roc_auc_score(Image_gt,Image_score)
    print("ROCAUC:",auc)
    
if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/VAD/SHTech', help='dataset path')
    parser.add_argument('--batch_size', type=int, default='40')
    parser.add_argument('--weights', type=str, default='result/checkpoint/best.pt', help='model.pt path(s)')
    parser.add_argument('--imsize', type=int, default=256, help='inference size (pixels)')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()

    print(args)
    main(args)