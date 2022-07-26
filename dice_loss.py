import torch
from torch.nn import functional as F

# label中数值应为0或1,pred中数值应在0~1之间
def binary_dice_coeff(pred:torch.Tensor, label:torch.Tensor,eval_mode:bool=False)->torch.Tensor:
    assert(pred.shape==label.shape)

    joint=(pred*label).sum()
    pred_size=pred.sum()
    label_size=label.sum()
    demoninator:int=pred_size+label_size
    if eval_mode==True:
        if(demoninator==0):
            return torch.ones(1)
        else:
            return (2*joint)/demoninator
    else:
        return (2*joint+0.0001)/(demoninator+0.0001)

def binary_dice_loss(pred:torch.Tensor, label:torch.Tensor,eval_mode:bool=False)->torch.Tensor:
    return 1-binary_dice_coeff(pred,label,eval_mode)

# def binary_focal_loss(pred:torch.Tensor, label:torch.Tensor, gamma:float=2)->torch.Tensor:
#     assert(pred.size()==label.size())
#     total:torch.Tensor=(0-label)*((1-pred)**gamma)*torch.log(pred)\
#         -(1-label)*(pred**gamma)*torch.log(pred)
#     weight:torch.Tensor=(1-label)*(pred**gamma)+label*((1-pred)**gamma)
#     return total.sum()/weight.sum()

def weightedBCE(pred:torch.Tensor, label:torch.Tensor,\
    target_weight:float=0.5):
    assert(pred.size()==label.size())
    assert(target_weight<1)
    ce_elem_wise:torch.Tensor=F.binary_cross_entropy(pred,label,reduction='none')
    
    weight_elem_wise:torch.Tensor=label*target_weight+(1-label)*(1-target_weight)
    total_loss:torch.Tensor=(ce_elem_wise*weight_elem_wise).sum()
    total_weight=weight_elem_wise.sum()
    return total_loss/total_weight

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
