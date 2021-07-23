import cv2, os, tkinter, natsort
import numpy as np
from tkinter import filedialog

def get_bound(img):
    X=[]
    def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            X.append([x,y])
            cv2.circle(img, (x, y), 2, (0, 0, 255), thickness=-1)
            cv2.imshow("click bound(press 'q' to stop and genarate mask)", img)

    cv2.namedWindow("click bound(press 'q' to stop and genarate mask)")
    cv2.setMouseCallback("click bound(press 'q' to stop and genarate mask)", on_EVENT_LBUTTONDOWN)
    cv2.imshow("click bound(press 'q' to stop and genarate mask)", img)
    cv2.waitKey(0)
    return np.array(X,np.int32)

def draw_mask(img_path='Image/1.jpg', Height=600, compression=1, inverse=False):
    img = cv2.imread(img_path)
    origin_shape=(int(compression*img.shape[1]),int(compression*img.shape[0]))
    Weight=int((img.shape[1]/img.shape[0])*Height)
    img = cv2.resize(img,(Weight,Height))
    
    X=get_bound(img)
    im=np.zeros_like(img, dtype="uint8")
    cor_xy = X.reshape((-1,1,2,))
    cv2.polylines(im, np.int32([cor_xy]), 1, 1)
    cv2.fillPoly(im, np.int32([cor_xy]), (255,255,255))
    im=cv2.resize(im,origin_shape)
    if inverse:
        im=255-im
    os.makedirs('./mask/', exist_ok=True)
    cv2.imwrite('./mask/{}_mask.jpg'.format(img_path.replace('\\','/').split('/')[-1][0:-4]),im)
    
if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()
    img_path = filedialog.askdirectory(**{'title':'select your img folder'})
    if not os.path.exists(img_path):
        print('selected path not exist!')
    else:
        filelist = natsort.natsorted(os.listdir(img_path))
        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.png'):
                filepath=os.path.join(os.path.abspath(img_path), item)
                draw_mask(img_path=filepath)

