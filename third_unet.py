import torch
from torch import nn
from typing import Union,Tuple

class DoubleConv(nn.Module):
    def __init__(self, in_ch:int, out_ch:int):
        super(DoubleConv, self).__init__()
        mid:int=(in_ch+out_ch)//2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid,kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch,kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

class NestedUNet(nn.Module):
    def __init__(self,in_channel:int,out_channel:int,use_deepsupervision:bool=False):
        super().__init__()

        self.use_deepsupervision=use_deepsupervision

        FILTERS = (32, 64, 128, 256, 512,)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_channel, FILTERS[0])
        self.conv1_0 = DoubleConv(FILTERS[0], FILTERS[1])
        self.conv2_0 = DoubleConv(FILTERS[1], FILTERS[2])
        self.conv3_0 = DoubleConv(FILTERS[2], FILTERS[3])
        self.conv4_0 = DoubleConv(FILTERS[3], FILTERS[4])

        self.conv0_1 = DoubleConv(FILTERS[0]+FILTERS[1], FILTERS[0])
        self.conv1_1 = DoubleConv(FILTERS[1]+FILTERS[2], FILTERS[1])
        self.conv2_1 = DoubleConv(FILTERS[2]+FILTERS[3], FILTERS[2])
        self.conv3_1 = DoubleConv(FILTERS[3]+FILTERS[4], FILTERS[3])

        self.conv0_2 = DoubleConv(FILTERS[0]*2+FILTERS[1], FILTERS[0])
        self.conv1_2 = DoubleConv(FILTERS[1]*2+FILTERS[2], FILTERS[1])
        self.conv2_2 = DoubleConv(FILTERS[2]*2+FILTERS[3], FILTERS[2])

        self.conv0_3 = DoubleConv(FILTERS[0]*3+FILTERS[1], FILTERS[0])
        self.conv1_3 = DoubleConv(FILTERS[1]*3+FILTERS[2], FILTERS[1])

        self.conv0_4 = DoubleConv(FILTERS[0]*4+FILTERS[1], FILTERS[0])
        self.sigmoid = nn.Sigmoid()
        if self.use_deepsupervision==True:
            self.final1 = nn.Conv2d(FILTERS[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(FILTERS[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(FILTERS[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(FILTERS[0], out_channel, kernel_size=1)
        else:
            self.final = nn.Conv2d(FILTERS[0], out_channel, kernel_size=1)


    def forward(self, input:torch.Tensor)->Union[torch.Tensor,Tuple[torch.Tensor],]:
        """
        返回值: 介于0~1

        """
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.use_deepsupervision==True:
            output1 = self.final1(x0_1)
            output1 = self.sigmoid(output1)
            output2 = self.final2(x0_2)
            output2 = self.sigmoid(output2)
            output3 = self.final3(x0_3)
            output3 = self.sigmoid(output3)
            output4 = self.final4(x0_4)
            output4 = self.sigmoid(output4)
            return (output1, output2, output3, output4,)

        else:
            output = self.final(x0_4)
            output = self.sigmoid(output)
            return output

if __name__ == '__main__':
    model=NestedUNet(1,1,False)

    model.train()
    model.zero_grad()

    x=torch.randn(size=(2,1,256,256))
    output:torch.Tensor
    output=model.forward(x)
    label:torch.Tensor = torch.randint(0,2,size=(2, 1,256,256,))
    

    from dice_loss import binary_dice_loss

    loss=binary_dice_loss(output,label)

    print('No gradient yet:',model.conv1_1.conv[0].weight.grad)
    loss.backward()
    print('Calculated gradient:', model.conv1_1.conv[0].weight.grad)

    print(output.type(),loss.item())
    print(output.size(),output.dim())

    def contain01(x:torch.Tensor)->bool:
        count1=(x>1).sum().item()
        count2=(x<0).sum().item()
        if count1==0 and count2==0:
            return True
        else:
            return False
    print('check01:',contain01(output))

    print('---------deep supervision----------')
    model=NestedUNet(1,1,True)

    model.train()
    model.zero_grad()


    output1:torch.Tensor
    output2:torch.Tensor
    output3:torch.Tensor
    output4:torch.Tensor
    output1,output2,output3,output4=model.forward(x)

    loss=binary_dice_loss(output1,label)+binary_dice_loss(output2,label)\
        +binary_dice_loss(output3,label)+binary_dice_loss(output4,label)

    print('No gradient yet:',model.conv1_1.conv[0].weight.grad)
    loss.backward()
    print('Calculated gradient:', model.conv1_1.conv[0].weight.grad)

    # print(output.type(),loss.item())
    # print(output.size(),output.dim())

    print('check01:',contain01(output1),contain01(output2),contain01(output3),contain01(output4))