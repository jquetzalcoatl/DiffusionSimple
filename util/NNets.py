import torch.nn as nn
import torch
import torch.nn.functional as F


class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConv2d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Pad input tensor with periodic boundary conditions
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        # Apply convolution
        x = self.conv(x)
        return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        sh = (x.shape[0],) + self.shape
        return x.view(sh)


class SimpleClas(nn.Module):
    def __init__(self):
        super(SimpleClas, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout2d(0.1),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   # nn.AvgPool2d(2), # USE for IMAGES 128x128
                                   nn.Conv2d(2048, 4096, 2, 1, 0),  # USE for IMAGES 128x128

                                   nn.Flatten(),

                                   nn.Linear(4096, 512),
                                   nn.Linear(512, 10),
                                   nn.LogSoftmax(dim=1))  #

    def forward(self, x):
        x = self.seqIn(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   # nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02))

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        return x1
    
# nn.Conv2d(3, 6, 3, 1, 1, padding_mode='reflect')
# pad = nn.ReflectionPad2d(1)
class SimpleCNNReflect(nn.Module):
    def __init__(self):
        super(SimpleCNNReflect, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, padding_mode='reflect'),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1, padding_mode='reflect'),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1, padding_mode='reflect'),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1, padding_mode='reflect'),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1, padding_mode='reflect'),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1, padding_mode='reflect'),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   # nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.Conv2d(2048, 1024, 3, 1, 1, padding_mode='reflect'),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.Conv2d(1024, 512, 3, 1, 1, padding_mode='reflect'),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.Conv2d(512, 256, 3, 1, 1, padding_mode='reflect'),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.Conv2d(256, 128, 3, 1, 1, padding_mode='reflect'),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.Conv2d(128, 64, 3, 1, 1, padding_mode='reflect'),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.Conv2d(64, 32, 3, 1, 1, padding_mode='reflect'),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.Conv2d(32, 1, 3, 1, 1, padding_mode='reflect'),
                                    nn.LeakyReLU(negative_slope=0.02))

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        return x1


class SimpleCNNJules(nn.Module):
    def __init__(self):
        super(SimpleCNNJules, self).__init__()
        self.nn1 = nn.Sequential(nn.Conv2d(3, 4, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.Dropout2d(self.dout1),
                                 nn.BatchNorm2d(4),
                                 
                                 nn.Conv2d(4, 8, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.BatchNorm2d(8),
                                 
                                 nn.Conv2d(8, 16, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.BatchNorm2d(16),
                                 
                                 nn.Conv2d(16, 8, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.BatchNorm2d(8),
                                 
                                 nn.Conv2d(8, 4, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.BatchNorm2d(4),
                                 
                                 nn.Conv2d(4, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.Dropout2d(self.dout2),
                                 nn.BatchNorm2d(1),
                                )
        self.seqIn = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   # nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02))
        self.seqLast = nn.Sequential(
                            nn.Conv2d(2, 1, 3, 1, 1),
                            nn.LeakyReLU(negative_slope=0.02)        
                        )

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        xnn1 = self.nn1(x)
        x2 = self.seqLast(torch.cat((xnn1, x1), dim=1))
        return x2
    

class SimpleCNNJulesPB(nn.Module):
    def __init__(self):
        super(SimpleCNNJulesPB, self).__init__()
        self.nn1 = nn.Sequential(PeriodicConv2d(3, 4, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.Dropout2d(self.dout1),
                                 nn.BatchNorm2d(4),
                                 
                                 PeriodicConv2d(4, 8, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.BatchNorm2d(8),
                                 
                                 PeriodicConv2d(8, 16, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.BatchNorm2d(16),
                                 
                                 PeriodicConv2d(16, 8, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.BatchNorm2d(8),
                                 
                                 PeriodicConv2d(8, 4, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.BatchNorm2d(4),
                                 
                                 PeriodicConv2d(4, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
#                                  nn.Dropout2d(self.dout2),
                                 nn.BatchNorm2d(1),
                                )
        self.seqIn = nn.Sequential(PeriodicConv2d(3, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   PeriodicConv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   PeriodicConv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   PeriodicConv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   PeriodicConv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   PeriodicConv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   # nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02))
        self.seqLast = nn.Sequential(
                            PeriodicConv2d(2, 1, 3, 1, 1),
                            nn.LeakyReLU(negative_slope=0.02)        
                        )

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        xnn1 = self.nn1(x)
        x2 = self.seqLast(torch.cat((xnn1, x1), dim=1))
        return x2
    
    
class SimpleCNNCat(nn.Module):
    def __init__(self):
        super(SimpleCNNCat, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   # nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02))
        self.seqLast = nn.Sequential(
                            nn.Conv2d(4, 1, 3, 1, 1),
                            nn.LeakyReLU(negative_slope=0.02)        
                        )

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        x2 = self.seqLast(torch.cat((x, x1), dim=1))
        return x2
    

class SimpleCNNCatPB(nn.Module):
    def __init__(self):
        super(SimpleCNNCatPB, self).__init__()
        self.seqIn = nn.Sequential(PeriodicConv2d(3, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   PeriodicConv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   PeriodicConv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   PeriodicConv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   PeriodicConv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   PeriodicConv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   # nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02))
        self.seqLast = nn.Sequential(
                            PeriodicConv2d(4, 1, 3, 1, 1),
                            nn.LeakyReLU(negative_slope=0.02)        
                        )

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        x2 = self.seqLast(torch.cat((x, x1), dim=1))
        return x2


class SimpleCNNConvT(nn.Module):
    def __init__(self):
        super(SimpleCNNConvT, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   # nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2),
                                    nn.ConvTranspose2d(128, 128, 2, 2, 0),

                                    nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2),
                                    nn.ConvTranspose2d(64, 64, 2, 2, 0),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2),
                                    nn.ConvTranspose2d(32, 32, 2, 2, 0),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02))

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        return x1


class SimpleCNN_L(nn.Module):
    def __init__(self):
        super(SimpleCNN_L, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   
                                   nn.Conv2d(2048, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   
                                   nn.Conv2d(2048, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
#                                    nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 2048, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    
                                    nn.ConvTranspose2d(2048, 2048, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),
                                    
                                    nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02))

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        return x1


class SimpleCNN_S(nn.Module):
    def __init__(self):
        super(SimpleCNN_S, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
#                                    nn.AvgPool2d(2),

#                                    nn.Conv2d(1024, 2048, 3, 1, 1),
#                                    nn.LeakyReLU(negative_slope=0.02),
                                   # nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(
#                                     nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
#                                     nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02))

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        return x1
    
    #nn.ConvTranspose2d(32, 1, 2, 2, 0)
class SimpleCNN_5K(nn.Module):
    def __init__(self):
        super(SimpleCNN_5K, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 5, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
#                                    nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(256, 128, 3, 1, 0),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(128, 64, 3, 1, 0),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 0),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 5, 1, 0),
                                    nn.LeakyReLU(negative_slope=0.02))

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        return x1


class SimpleCNN_10K(nn.Module):
    def __init__(self):
        super(SimpleCNN_10K, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 10, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 10, 1, 1),
#                                    nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 10, 1, 1),
#                                    nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 10, 1, 1),
#                                    nn.BatchNorm2d(512),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 10, 1, 1),
#                                    nn.BatchNorm2d(1024),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 10, 1, 1),
#                                    nn.BatchNorm2d(2048),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.Upsample(scale_factor=2),
                                    nn.ConvTranspose2d(2048, 1024, 10, 1, 1),
#                                     nn.BatchNorm2d(1024),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(1024, 512, 10, 1, 1),
#                                     nn.BatchNorm2d(512),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(512, 256, 10, 1, 1),
#                                     nn.BatchNorm2d(256),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(256, 128, 10, 1, 1),
#                                     nn.BatchNorm2d(128),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(128, 64, 10, 1, 1),
#                                     nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 10, 1, 1),
#                                     nn.BatchNorm2d(32),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 10, 1, 1),
#                                     nn.BatchNorm2d(1),
                                    nn.LeakyReLU(negative_slope=0.02)
                                   )

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        return x1

    
class SimpleCNN100(nn.Module):
    def __init__(self): #Same as previous but specifically for 100x100 input
        super(SimpleCNN100, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
#                                    nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2), #6

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),  #12

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2), #24

#                                     nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.ConvTranspose2d(128,64,2,1,0),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02))

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        return x1
    
class JuliaCNN100(nn.Module):
    def __init__(self, dout1=0.4, dout2=0.4, dout3=0.1, dout4=0.1, p1=0.5):
        super(JuliaCNN100, self).__init__() #pytorch version of the Julia network
        self.dout1 = dout1
        self.dout2 = dout2
        self.dout3 = dout3
        self.dout4 = dout4
        self.p1 = p1
        self.p2 = 1.0 - p1
        self.nn1 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.Dropout2d(self.dout1),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.ReLU(),
                                 nn.Dropout2d(self.dout2),
                                 nn.BatchNorm2d(1),
                                )
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout2d(self.dout3),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),
#                                    nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 2, 0),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 2, 0),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2), #6

                                    nn.ConvTranspose2d(512, 256, 3, 2, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2),  #12

                                    nn.ConvTranspose2d(256, 128, 3, 2, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2), #24

#                                     nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.ConvTranspose2d(128,64,3,2,0),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2),

                                    nn.Dropout2d(self.dout4),
                                    nn.ConvTranspose2d(64, 1, 4, 2, 2),
                                    nn.ReLU(),
#                                     nn.BatchNorm2d(1),

                                   )
        self.nn2 = nn.Sequential(self.seqIn,
                                 self.seqOut,        
                                )
        
    def updatePs(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def forward(self, x):
        x1 = self.nn1(x)
        x2 = self.nn2(x)
        xout = self.p1 * x1 + self.p2 * x2
        return xout
    
class JuliaCNN100_2(nn.Module):
    def __init__(self, dout1=0.4, dout2=0.4, dout3=0.1, dout4=0.1, p1=0.5):
        super(JuliaCNN100_2, self).__init__() #pytorch version of the Julia network
        self.dout1 = dout1
        self.dout2 = dout2
        self.dout3 = dout3
        self.dout4 = dout4
        self.p1 = p1
        self.p2 = 1.0 - p1
        self.nn1 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.Dropout2d(self.dout1),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.ReLU(),
                                 nn.Dropout2d(self.dout2),
                                 nn.BatchNorm2d(1),
                                )
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
#                                    nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2), #6

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),  #12

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2), #24

#                                     nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.ConvTranspose2d(128,64,2,1,0),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02)
                                   )
        
        self.nn2 = nn.Sequential(self.seqIn,
                                 self.seqOut,        
                                )
        
    def updatePs(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def forward(self, x):
        x1 = self.nn1(x)
        x2 = self.nn2(x)
        xout = self.p1 * x1 + self.p2 * x2
        return xout


class DNN(nn.Module):
    def __init__(self, p1=1.0, p2=0.0, dout1=0.1, dout2=0.1):
        super(DNN, self).__init__()
        self.p1 = p1
        self.p2 = p2
        self.dout1 = dout1
        self.dout2 = dout2
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Dropout2d(self.dout1),
                                   nn.BatchNorm2d(64),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(4),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Dropout2d(self.dout2),
                                   nn.ConvTranspose2d(64, 1, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.0),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.ReLU(),
                                   # nn.BatchNorm2d(1),

                                   )

    def forward(self, x):
        x = self.p1 * self.seqIn(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.blk1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        
        self.blkUp1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.upConv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
        )
        self.upConv3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, 2, 2, 0)
        )
        self.upConv4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2, 0)
        )
        self.lastlayer = nn.ConvTranspose2d(64, 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.blk1(x) #512
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        xfinal = self.lastlayer(x9)

        return xfinal


class UNetBias0(nn.Module):
    def __init__(self):
        super(UNetBias0, self).__init__()
        self.blk1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        
        self.blkUp1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.upConv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ConvTranspose2d(1024, 512, 2, 2, 0, bias=False)
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
        )
        self.upConv3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, 2, 2, 0, bias=False)
        )
        self.upConv4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2, 0, bias=False)
        )
        self.lastlayer = nn.ConvTranspose2d(64, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        x1 = self.blk1(x) #512
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        xfinal = self.lastlayer(x9)

        return xfinal
    

class UNetBias0PB(nn.Module):
    def __init__(self):
        super(UNetBias0PB, self).__init__()
        self.blk1 = nn.Sequential(
            PeriodicConv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk2 = nn.Sequential(
            PeriodicConv2d(64, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(128, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk3 = nn.Sequential(
            PeriodicConv2d(128, 256, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(256, 256, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk4 = nn.Sequential(
            PeriodicConv2d(256, 512, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(512, 512, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk5 = nn.Sequential(
            PeriodicConv2d(512, 1024, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        
        self.blkUp1 = nn.Sequential(
            PeriodicConv2d(1024, 512, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(512, 512, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp2 = nn.Sequential(
            PeriodicConv2d(512, 256, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(256, 256, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp3 = nn.Sequential(
            PeriodicConv2d(256, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

           PeriodicConv2d(128, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp4 = nn.Sequential(
            PeriodicConv2d(128, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.upConv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ConvTranspose2d(1024, 512, 2, 2, 0, bias=False)
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
        )
        self.upConv3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, 2, 2, 0, bias=False)
        )
        self.upConv4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2, 0, bias=False)
        )
        self.lastlayer = nn.ConvTranspose2d(64, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        x1 = self.blk1(x) #512
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        xfinal = self.lastlayer(x9)

        return xfinal
    

class UNetPrelu(nn.Module):
    def __init__(self):
        super(UNetPrelu, self).__init__()
        self.blk1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.PReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
        )
        self.blk2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.PReLU(),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.PReLU(),
        )
        self.blk3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.PReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.PReLU(),
        )
        self.blk4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.PReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.PReLU(),
        )
        self.blk5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.PReLU(),

            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.PReLU(),
        )
        
        self.blkUp1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.PReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.PReLU(),
        )
        self.blkUp2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.PReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.PReLU(),
        )
        self.blkUp3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.PReLU(),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.PReLU(),
        )
        self.blkUp4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.PReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
        )
        self.upConv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
        )
        self.upConv3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, 2, 2, 0)
        )
        self.upConv4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2, 0)
        )
        self.lastlayer = nn.ConvTranspose2d(64, 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.blk1(x) #512
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        xfinal = self.lastlayer(x9)

        return xfinal
    
    
class UNetPreluPB(nn.Module):
    def __init__(self):
        super(UNetPreluPB, self).__init__()
        self.blk1 = nn.Sequential(
            PeriodicConv2d(3, 64, 3, 1, 1),
            nn.PReLU(),

            PeriodicConv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
        )
        self.blk2 = nn.Sequential(
            PeriodicConv2d(64, 128, 3, 1, 1),
            nn.PReLU(),

            PeriodicConv2d(128, 128, 3, 1, 1),
            nn.PReLU(),
        )
        self.blk3 = nn.Sequential(
            PeriodicConv2d(128, 256, 3, 1, 1),
            nn.PReLU(),

            PeriodicConv2d(256, 256, 3, 1, 1),
            nn.PReLU(),
        )
        self.blk4 = nn.Sequential(
            PeriodicConv2d(256, 512, 3, 1, 1),
            nn.PReLU(),

            PeriodicConv2d(512, 512, 3, 1, 1),
            nn.PReLU(),
        )
        self.blk5 = nn.Sequential(
            PeriodicConv2d(512, 1024, 3, 1, 1),
            nn.PReLU(),

            PeriodicConv2d(1024, 1024, 3, 1, 1),
            nn.PReLU(),
        )
        
        self.blkUp1 = nn.Sequential(
            PeriodicConv2d(1024, 512, 3, 1, 1),
            nn.PReLU(),

            PeriodicConv2d(512, 512, 3, 1, 1),
            nn.PReLU(),
        )
        self.blkUp2 = nn.Sequential(
            PeriodicConv2d(512, 256, 3, 1, 1),
            nn.PReLU(),

            PeriodicConv2d(256, 256, 3, 1, 1),
            nn.PReLU(),
        )
        self.blkUp3 = nn.Sequential(
            PeriodicConv2d(256, 128, 3, 1, 1),
            nn.PReLU(),

            PeriodicConv2d(128, 128, 3, 1, 1),
            nn.PReLU(),
        )
        self.blkUp4 = nn.Sequential(
            PeriodicConv2d(128, 64, 3, 1, 1),
            nn.PReLU(),

            PeriodicConv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
        )
        self.upConv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
        )
        self.upConv3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, 2, 2, 0)
        )
        self.upConv4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2, 0)
        )
        self.lastlayer = nn.ConvTranspose2d(64, 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.blk1(x) #512
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        xfinal = self.lastlayer(x9)

        return xfinal
    
    
class UNetPrelu2PB(nn.Module):
    def __init__(self):
        super(UNetPrelu2PB, self).__init__()
        self.blk1 = nn.Sequential(
            PeriodicConv2d(3, 64, 3, 1, 1),
            nn.PReLU(64, -0.02),

            PeriodicConv2d(64, 64, 3, 1, 1),
            nn.PReLU(64, -0.02),
        )
        self.blk2 = nn.Sequential(
            PeriodicConv2d(64, 128, 3, 1, 1),
            nn.PReLU(128, -0.02),

            PeriodicConv2d(128, 128, 3, 1, 1),
            nn.PReLU(128, -0.02),
        )
        self.blk3 = nn.Sequential(
            PeriodicConv2d(128, 256, 3, 1, 1),
            nn.PReLU(256, -0.02),

            PeriodicConv2d(256, 256, 3, 1, 1),
            nn.PReLU(256, -0.02),
        )
        self.blk4 = nn.Sequential(
            PeriodicConv2d(256, 512, 3, 1, 1),
            nn.PReLU(512, -0.02),

            PeriodicConv2d(512, 512, 3, 1, 1),
            nn.PReLU(512, -0.02),
        )
        self.blk5 = nn.Sequential(
            PeriodicConv2d(512, 1024, 3, 1, 1),
            nn.PReLU(1024, -0.02),

            PeriodicConv2d(1024, 1024, 3, 1, 1),
            nn.PReLU(1024, -0.02),
        )
        
        self.blkUp1 = nn.Sequential(
            PeriodicConv2d(1024, 512, 3, 1, 1),
            nn.PReLU(512, -0.02),

            PeriodicConv2d(512, 512, 3, 1, 1),
            nn.PReLU(512, -0.02),
        )
        self.blkUp2 = nn.Sequential(
            PeriodicConv2d(512, 256, 3, 1, 1),
            nn.PReLU(256, -0.02),

            PeriodicConv2d(256, 256, 3, 1, 1),
            nn.PReLU(256, -0.02),
        )
        self.blkUp3 = nn.Sequential(
            PeriodicConv2d(256, 128, 3, 1, 1),
            nn.PReLU(128, -0.02),

            PeriodicConv2d(128, 128, 3, 1, 1),
            nn.PReLU(128, -0.02),
        )
        self.blkUp4 = nn.Sequential(
            PeriodicConv2d(128, 64, 3, 1, 1),
            nn.PReLU(64, -0.02),

            PeriodicConv2d(64, 64, 3, 1, 1),
            nn.PReLU(64, -0.02),
        )
        self.upConv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
        )
        self.upConv3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, 2, 2, 0)
        )
        self.upConv4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2, 0)
        )
        self.lastlayer = nn.ConvTranspose2d(64, 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.blk1(x) #512
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        xfinal = self.lastlayer(x9)

        return xfinal
    
    
class LeakyUNet(nn.Module):
    def __init__(self):
        super(LeakyUNet, self).__init__()
        self.blk1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blk2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blk3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blk4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blk5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        
        self.blkUp1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blkUp2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blkUp3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blkUp4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.upConv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
        )
        self.upConv3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, 2, 2, 0)
        )
        self.upConv4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2, 0)
        )
        self.lastlayer = nn.Sequential( nn.ConvTranspose2d(64, 1, 3, 1, 1),
                                       nn.LeakyReLU(negative_slope=0.02),
                                      )

    def forward(self, x):
        x1 = self.blk1(x) #512
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        xfinal = self.lastlayer(x9)

        return xfinal
    
    
class LeakyUNetPB(nn.Module):
    def __init__(self):
        super(LeakyUNetPB, self).__init__()
        self.blk1 = nn.Sequential(
            PeriodicConv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            PeriodicConv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blk2 = nn.Sequential(
            PeriodicConv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            PeriodicConv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blk3 = nn.Sequential(
            PeriodicConv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            PeriodicConv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blk4 = nn.Sequential(
            PeriodicConv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            PeriodicConv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blk5 = nn.Sequential(
            PeriodicConv2d(512, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            PeriodicConv2d(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        
        self.blkUp1 = nn.Sequential(
            PeriodicConv2d(1024, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            PeriodicConv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blkUp2 = nn.Sequential(
            PeriodicConv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            PeriodicConv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blkUp3 = nn.Sequential(
            PeriodicConv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            PeriodicConv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.blkUp4 = nn.Sequential(
            PeriodicConv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),

            PeriodicConv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.02),
        )
        self.upConv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
        )
        self.upConv3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, 2, 2, 0)
        )
        self.upConv4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2, 0)
        )
        self.lastlayer = nn.Sequential( nn.ConvTranspose2d(64, 1, 3, 1, 1),
                                       nn.LeakyReLU(negative_slope=0.02),
                                      )

    def forward(self, x):
        x1 = self.blk1(x) #512
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        xfinal = self.lastlayer(x9)

        return xfinal


class UNetPB(nn.Module):
    def __init__(self):
        super(UNetPB, self).__init__()
        self.blk1 = nn.Sequential(
            PeriodicConv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk2 = nn.Sequential(
            PeriodicConv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk3 = nn.Sequential(
            PeriodicConv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk4 = nn.Sequential(
            PeriodicConv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk5 = nn.Sequential(
            PeriodicConv2d(512, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        
        self.blkUp1 = nn.Sequential(
            PeriodicConv2d(1024, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp2 = nn.Sequential(
            PeriodicConv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp3 = nn.Sequential(
            PeriodicConv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp4 = nn.Sequential(
            PeriodicConv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            PeriodicConv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.upConv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
        )
        self.upConv3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(256, 128, 2, 2, 0)
        )
        self.upConv4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2, 0)
        )
        self.lastlayer = nn.ConvTranspose2d(64, 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.blk1(x) #512
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        xfinal = self.lastlayer(x9)

        return xfinal
    
import torch.nn as nn

class UNetGPT(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetGPT, self).__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
        )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
    
        self.output = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
#             nn.Sigmoid(),
            nn.ReLU(),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        center = self.center(enc4)
        dec4 = self.decoder4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=False)], 1))
        dec3 = self.decoder3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear', align_corners=False)], 1))
        dec2 = self.decoder2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=False)], 1))
        dec1 = self.decoder1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=False)], 1))
        output = self.output(dec1)
        return output




class DiffDisc(nn.Module):
    def __init__(self):
        super(DiffDisc, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(2, 64, 3, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout2d(0.1),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   
                                   nn.Conv2d(2048, 2048, 4, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Flatten(),

                                   nn.Linear(2048, 512),
                                   nn.Linear(512, 1),
                                   nn.Sigmoid(),
#                                    nn.LeakyReLU(negative_slope=0.02),
                                  )  #

    def forward(self, x):
        x = self.seqIn(x)
        return x
    
    
class SimpleDisc(nn.Module):
    def __init__(self):
        super(SimpleDisc, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout2d(0.1),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),

                                   nn.Flatten(),

                                   nn.Linear(2048, 512),
                                   nn.Linear(512, 1),
                                   nn.Sigmoid())  #

    def forward(self, x):
        x = self.seqIn(x)
        return x


class SimpleGen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.seqIn = nn.Sequential(nn.Linear(100, 160000),
                                   nn.BatchNorm1d(160000),
                                   nn.ReLU(),
                                   Reshape(256, 25, 25),
                                   nn.ConvTranspose2d(256, 128, 5, 2, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(128, 64, 4, 1, 2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(64, 1, 4, 2, 1),
                                   nn.Hardtanh()
                                   )  #

    def forward(self, x):
        x = self.seqIn(x)
        return x
