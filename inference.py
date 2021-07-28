import os,sys,natsort,argparse,time,subprocess,cv2
import torch,torchvision,einops
import numpy as np
from PIL import Image
from torchvision import transforms as T
from datetime import datetime
from multiprocessing import Pool
from model.pmas import PMAS

def ae_preprocess(img_path,im_size,device):
    img=Image.open(img_path).convert('RGB')
    trans=T.Compose([T.Resize((im_size,im_size), Image.ANTIALIAS),
                T.ToTensor(),T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    img=trans(img)
    img_tensor=img[np.newaxis,:]
    return img_tensor.to(device)

def merge_video(img_path):
    filelist = natsort.natsorted(os.listdir(img_path))
    img=cv2.imread(os.path.join(img_path,filelist[0]))
    img_size=img.shape
    fps = 25
    file_path = img_path +'_video' + ".avi"
    fourcc = cv2.VideoWriter_fourcc('P','I','M','1')
    video = cv2.VideoWriter( file_path, fourcc, fps ,(img_size[1],img_size[0]))
    for item in filelist:
        if item.endswith('.jpg') or item.endswith('.png'):
            item = os.path.join(img_path,item)
            img = cv2.imread(item)
            video.write(img)        
    video.release()

def main(args):
    print("=> main task started: {}".format(datetime.now().strftime('%H:%M:%S')))
    
    # * load model
    a=time.time()
    model=PMAS().to(args.device)
    model.load_state_dict(torch.load(args.weights)[0])
    model.eval()
    b=time.time()
    print("=> load model, cost:{:.2f}s".format(b-a))
    
    # * clean output folder
    sys_cmd="rm -rf {}".format(args.output)
    child = subprocess.Popen(sys_cmd,shell=True)
    child.wait()
    os.makedirs(args.output,exist_ok=True)
    c=time.time()
    print("=> clean the output path, cost:{:.2f}s".format(c-b))
    
    # * multi process
    if args.pools > 1: 
        myP = Pool(args.pools)
        print("=> using process pool")
    else:
        myP=None
        print("=> using single process")

    # * load image and process
    img_List=natsort.natsorted(os.listdir(args.input))
    total_num=len(img_List)
    with torch.no_grad():
        for index,img_name in enumerate(img_List):
            x=ae_preprocess(os.path.join(args.input,img_name),args.imsize,args.device)
            mask,_=model(x)
            mask=einops.repeat(mask, 'b c h w -> b (repeat c) h w', repeat=3)
            result = torch.cat([x, mask], 0)
            torchvision.utils.save_image(result,
                    os.path.join(args.output,img_name),
                    nrow=2,
                    normalize=True,
                    range=(-1, 1),)
            sys.stdout.write("\r=> processing at %d; total: %d" %(index, total_num))
            sys.stdout.flush()

    if args.pools > 1:
        myP.close()
        myP.join()
    print("\n=> process done {}/{} images, total cost: {:.2f}s [{:.2f} fps]".format(len(os.listdir(args.output)),total_num,time.time()-c,len(os.listdir(args.output))/(time.time()-c)))
    
    # * merge video
    if args.video:
        print("=> generating video, may take some times")
        merge_video(args.output)
        
    print("=> main task finished: {}".format(datetime.now().strftime('%H:%M:%S')))
    
if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="test_imgs/input", help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="test_imgs/output", help='folder to save result imgs, can not use input folder')
    parser.add_argument('--pools',type=int, default=1, help='max pool num')
    parser.add_argument('--video', action='store_true', help='save result to video')

    parser.add_argument('--weights', type=str, default='result/checkpoint/best.pt', help='model.pt path(s)')
    parser.add_argument('--imsize', type=int, default=256, help='inference size (pixels)')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()
    print(args)
    main(args)

