import numpy as np
import matplotlib.pyplot as plt
import pickle

with open(r"result/eval_result.pkl", "rb") as f:    
    Img_gt,Img_score=pickle.load(f)

plt.figure(figsize=(12,5))
plt.plot(np.arange(0,len(Img_score),1),Img_score)
plt.plot(np.arange(0,len(Img_score),1),Img_gt[0:len(Img_score)])
plt.xticks(np.arange(0,len(Img_score),100))
plt.savefig('result/plot_auc.jpg', dpi=100)