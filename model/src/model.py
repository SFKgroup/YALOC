import numpy as np
from mindspore import dtype as mstype
from mindspore import nn, ops, Tensor



class CBL(nn.Cell):
    def __init__(self, in_channels, out_channels=-1,kernel_size=3,stride=1,padding=1):
        if out_channels == -1:
            out_channels = in_channels
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=stride,
                              pad_mode = 'pad',
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.active = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.active(x)
        return x

if __name__ == '__main__':
    model = CBL(3)
    data = Tensor(np.random.randn(3, 3, 224, 224), dtype=mstype.float32)
    print(model(data).shape)
    

class Resunit_common(nn.Cell):
    def __init__(self,channels):
        super().__init__()
        self.cbl1 = CBL(channels)
        self.cbl2 = CBL(channels)

    def construct(self,x):
        x1 = self.cbl1(x)
        x1 = self.cbl2(x1)
        x = x + x1
        return x
    
if __name__ == '__main__':
    model = Resunit_common(3)
    data = Tensor(np.random.randn(3, 3, 224, 224), dtype=mstype.float32)
    print(model(data).shape)
    

class Resunit_down(nn.Cell):
    def __init__(self,channels):
        super().__init__()
        self.cbl1 = CBL(channels,channels*2,stride=2)
        self.cbl2 = CBL(channels*2)
        self.conv = nn.Conv2d(channels,
                              channels*2,
                              kernel_size=1,
                              stride=2,
                              padding=0)
        
    def construct(self,x):
        x1 = self.cbl1(x)
        x1 = self.cbl2(x1)
        x = self.conv(x)
        x = x + x1
        return(x)

if __name__ == '__main__':
    model = Resunit_down(3)
    data = Tensor(np.random.randn(3, 3, 224, 224), dtype=mstype.float32)
    print(model(data).shape)

class ResNet(nn.Cell):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        k = 16
        self.conv1 = nn.Conv2d(3,
                              k,
                              kernel_size=7,
                              stride=2,
                              pad_mode = 'pad',
                              padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.resunit1 = Resunit_common(k)
        self.resunit2 = Resunit_common(k)
        self.resunit3 = Resunit_down(k)
        self.resunit4 = Resunit_common(2*k)
        self.resunit5 = Resunit_down(2*k)
        self.resunit6 = Resunit_common(4*k)
        self.resunit7 = Resunit_down(4*k)
        self.resunit8 = Resunit_common(8*k)
        self.pool2 = ops.ReduceMean()
        self.linear = nn.Dense(8*k, num_classes)
        
    def construct(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.resunit1(x)
        x = self.resunit2(x)
        x = self.resunit3(x)
        x = self.resunit4(x)
        x = self.resunit5(x)
        x = self.resunit6(x)
        x = self.resunit7(x)
        x = self.resunit8(x)
        x = self.pool2(x, (2, 3))
        x = self.linear(x)
        
        return x


def resnet(num_classes=1000):
    net = ResNet(num_classes=num_classes)
    return net


if __name__ == '__main__':
    model = resnet(21)
    data = Tensor(np.random.randn(3, 3, 224, 224), dtype=mstype.float32)
    print(model(data).shape)