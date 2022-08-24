import torch

# label中数值应为0或1,pred中数值应在0~1之间
def binary_dice_coeff(pred:torch.Tensor, label:torch.Tensor)->torch.Tensor:
    assert(pred.shape==label.shape)

    joint=(pred*label).sum()
    pred_size=pred.sum()
    label_size=label.sum()
    return 2*joint/(pred_size+label_size)

def binary_dice_loss(pred:torch.Tensor, label:torch.Tensor)->torch.Tensor:
    return 1-binary_dice_coeff(pred,label)

def binary_focal_loss(pred:torch.Tensor, label:torch.Tensor, gamma:float=2)->torch.Tensor:
    assert(pred.size()==label.size())
    total:torch.Tensor=(0-label)*((1-pred)**gamma)*torch.log(pred)\
        -(1-label)*(pred**gamma)*torch.log(pred)
    weight:torch.Tensor=(1-label)*(pred**gamma)+label*((1-pred)**gamma)
    return total.sum()/weight.sum()

def weightedBCE(pred:torch.Tensor, label:torch.Tensor,\
    target_weight:float=0.5):
    assert(pred.size()==label.size())
    assert(target_weight<1)
    total:torch.Tensor=0-label*target_weight*torch.log(0.0001+pred)\
        -(1-label)*(1-target_weight)*torch.log(1.0001-pred)
    total=total.sum()

    count1=label.sum()
    count0=label.numel()-count1
    total_weight:torch.Tensor=count0*(1-target_weight)+count1*target_weight
    return total/total_weight

if __name__ == '__main__':
    import attention_unet
    myModel=attention_unet.attention_Unet(1,1)

    img=torch.randn(size=(4,1,256,256))

    label=torch.randint(0,2,size=(4,1,256,256))
    label=label.float()
    print(label.dtype)

    
    
    optimizer = torch.optim.Adam(myModel.parameters())
    optimizer.zero_grad()

    pred=myModel(label)

    criterion=torch.nn.BCELoss()
    loss:torch.Tensor
    loss=binary_dice_loss(pred,label)+criterion(pred,label)
    loss.backward()
    optimizer.step()

    print('损失:',loss.item())

    tmp_arr:torch.Tensor=pred.view(-1)
    for x in tmp_arr:
        if x <0 or x>1:
            print('网络输出层越界')

    print('数值0-1检查')
