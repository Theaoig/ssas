import torch,torchvision,argparse,os,sys,einops
from torch.utils.data import DataLoader
from datetime import datetime

from model.pmas import PMAS
from dataset.dataset import PMASDataset
from eval import evaluation


def add_log(info,log_path):
    with open(log_path,'a',encoding='utf-8') as f:
            f.writelines(info)
                    
def Train(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ckpt_path=os.path.join(args.save_path,'checkpoint')
    sample_path=os.path.join(args.save_path,'train_samples')
    log_path=os.path.join(args.save_path,"train_log.log")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    
    add_log("========== Start Training at {} ==========\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),log_path)
    print(args)
    dataset = PMASDataset(dataset_path=args.dataset_path,imsize=args.imsize, phase="train", sparse=args.sparse)
    print("total {} imgs.".format(len(dataset)))
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=16)
    class_name = os.path.basename(args.dataset_path)

    model = PMAS().to(args.device)
    if os.path.exists(args.weights):
        weights,best_auc=torch.load(args.weights)
        model.load_state_dict(weights)
        add_log("=> reload weights from {}, auc: {} \n".format(args.weights,best_auc),log_path)
    else:
        best_auc=0
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    label_loss=torch.nn.BCELoss()
    mask_loss=torch.nn.SmoothL1Loss()
    for epoch in range(1,args.epochs+1):
        model.train()
        for i, (img,gt_mask,gt_label) in enumerate(loader):
            img=img.to(args.device)
            gt_mask=gt_mask.to(args.device)
            gt_label=gt_label.to(args.device)
            optimizer.zero_grad()
            
            pred_mask,pred_label=model(img)
            
            loss=mask_loss(pred_mask,gt_mask.detach())+args.lamdba_1*label_loss(pred_label,gt_label.detach())
            loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]["lr"]
            sys.stdout.write("\r=> iters: {} - {}, loss:{:.5f} - {:.5f}".format(epoch,i,mask_loss(pred_mask,gt_mask.detach()).item(),args.lamdba_1*label_loss(pred_label,gt_label.detach())))
            sys.stdout.flush()
            if i==0:
                B,*_=img.shape
                B=min(B,8)
                m1=einops.repeat(gt_mask, 'b c h w -> b (repeat c) h w', repeat=3)
                m2=einops.repeat(pred_mask, 'b c h w -> b (repeat c) h w', repeat=3)
                sample = torch.cat([img[:B], m1[:B], m2[:B]], 0)
                torchvision.utils.save_image(sample, 
                    os.path.join(sample_path,"{}_{}.jpg".format(class_name,epoch)),
                    nrow=B,
                    normalize=True,
                    range=(-1, 1),)
        print(" ")
        add_log("=> {}th epoch done, loss: {:.5f}, lr:{:.5f}\n".format(epoch,loss.item(),lr),log_path)
        torch.save([model.state_dict(),loss.item()], os.path.join(ckpt_path,"last.pt"))
        if epoch%5==0:
            model.eval()
            auc=evaluation(model,args.dataset_path,args.batch_size,args.imsize,args.device,False)
            add_log("eval auc: {} \n".format(auc),log_path)
            if auc > best_auc:
                best_auc = auc
                torch.save([model.state_dict(),auc], os.path.join(ckpt_path,"best.pt"))
        scheduler.step()
        
        
    print("{} traing process done".format(class_name))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default='32')
    parser.add_argument('--epochs', type=int, default='100')
    parser.add_argument('--dataset_path', type=str, default='/data/VAD/SHTech', help='dataset path')
    parser.add_argument('--sparse', type=int, default='10')
    parser.add_argument('--imsize', type=int, default='256')
    parser.add_argument('--save_path', type=str, default='result', help='path to save log and ckpt')
    parser.add_argument('--device', type=str, default='cuda', help='device number')
    parser.add_argument('--lr', type=float, default='1e-5', help='init learning rate')
    parser.add_argument('--lamdba_1', type=float, default='0.05')
    parser.add_argument('--weights', type=str, default='result/checkpoint/last.pt')
    args=parser.parse_args()

    Train(args)
    