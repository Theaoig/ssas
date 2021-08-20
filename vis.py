import numpy as np
import matplotlib.pyplot as plt
import pickle,os,cv2
from sklearn.metrics import roc_auc_score
from eval import normalize, moving_avg

with open(r"result/eval_result.pkl", "rb") as f:    
    Img_gt,Img_score=pickle.load(f)

left=0
right=0
test_folder=os.path.join("/data/VAD/SHTech","testing","frames")
for folder in sorted(os.listdir(test_folder)):
    length=len(os.listdir(os.path.join(test_folder,folder)))
    right=left+length
    
    mask=np.load("/data/VAD/SHTech/testing/test_pixel_mask/{}.npy".format(folder))
    Image_score=[]
    for i in range(mask.shape[0]):
        score=1 if mask[i,:,:].sum()>0 else 0
        Image_score.append(score)
    
    assert len(Image_score)==length, "wrong!"
    Img_score[left:right]=normalize(moving_avg(Img_score[left:right]))
    try:
        auc=roc_auc_score(Img_gt[left:right],Img_score[left:right])
        print("=> {} AUC: {:.3f}".format(folder,auc))
    except:
        if Img_gt[left]==0:
            auc=1-(np.array(Img_score[left:right]).mean())
        else:
            auc=np.array(Img_score[left:right]).mean()
        print("=> {} has one class, fake auc: {:.3f}".format(folder,auc))
    
    plt.figure(figsize=(10,5))
    plt.title("{} AUC: {:.3f}".format(folder,auc))
    plt.plot(np.arange(0,length,1),Img_score[left:right])
    plt.plot(np.arange(0,length,1),Img_gt[left:right])
    plt.xticks(np.arange(0,length,50))
    plt.savefig('result/plot_auc_{}.jpg'.format(folder), dpi=100)
    plt.close()
    left=right
    
auc=roc_auc_score(Img_gt,Img_score)
print("=> {} AUC: {:.3f}".format("total",auc))