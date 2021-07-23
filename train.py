from numpy.lib.npyio import load
import torch,torchvision,argparse,os
from torch.utils.data import DataLoader
from datetime import datetime

from einops import repeat
from model.PAS import PAS
from dataset.dataset import PASDataset


def Train(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ckpt_path=os.path.join(args.save_path,'checkpoint')
    sample_path=os.path.join(args.save_path,'train_samples')
    log_path=os.path.join(args.save_path,'train_log.log')
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    
    with open(log_path,'a',encoding='utf-8') as f:
        print(args,file=f)
        f.writelines("\n"+"========== Training log: ==========\n")
    dataset = PASDataset(dataset_path=args.dataset_path,imsize=256, sparse=90)
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=16)
    class_name = os.path.basename(args.dataset_path)

    model = PAS().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    label_loss=torch.nn.BCELoss()
    mask_loss=torch.nn.L1Loss()
    
    for epoch in range(args.epochs):
        model.train()
        for i, (img,gt_mask,gt_label) in enumerate(loader):
            optimizer.zero_grad()
            pred_label,pred_mask=model(img.to(args.device))
            
            loss=label_loss(pred_label,gt_label.to(args.device).detach())+mask_loss(pred_mask,gt_mask.to(args.device).detach())
            loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]["lr"]
            if i==0:
                B,*_=img.shape
                print(img.shape,gt_mask.shape)
                gt_mask=repeat(gt_mask, 'b c h w -> b (re c) h w', re=3)
                sample = torch.cat([img, gt_mask], 0)
                torchvision.utils.save_image(sample, 
                    os.path.join(sample_path,"{}_{}.jpg".format(class_name,epoch+1)),
                    nrow=B,
                    normalize=True,
                    range=(-1, 1),)
        scheduler.step()
        
    print("{} traing process done".format(class_name))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default='8')
    parser.add_argument('--epochs', type=int, default='1')
    parser.add_argument('--dataset_path', type=str, default='dataset/SHTech', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save log and ckpt')
    parser.add_argument('--device', type=str, default='cuda', help='device number')
    parser.add_argument('--init_lr', type=float, default='2e-4', help='init learning rate')
    args=parser.parse_args()

    Train(args)
    