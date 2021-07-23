import torch

class ChannelAttention(torch.nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
           
        self.fc = torch.nn.Sequential(torch.nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               torch.nn.ReLU(),
                               torch.nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class CBAM(torch.nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca=ChannelAttention(in_channels)
        self.sa=SpatialAttention()
        
    def forward(self,x):
        out=self.ca(x)*x
        out=self.sa(out)*out
        return out
        
    
if __name__ =="__main__":
    fmap=torch.ones((1,256,80,80))
    fmap[:,0,:,:]=0.1
    print(fmap[:,0,:,:])
    _,channels,*_=fmap.shape
    net=CBAM(channels)
    out=net(fmap)
    print(fmap[:,0,:,:])
    
