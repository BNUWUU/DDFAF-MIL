import DDFAMIL
import torch
from torchvision import transforms
from MyLoader import OrigDataset as XDataset
from utils import inference, train, group_argtopk, writecsv, group_max, calc_err, tfpn, score, FocalLoss, BCE
from torch import nn
import numpy as np
from utils import inference_info

norm = 'BN'
ckpt_path = 'ckpt_acc_100_1_5_1750941644.pth'
size = 100

model = DDFAMIL.DualMILNet()

checkpoint=torch.load(ckpt_path)
# best_f1 = ckpt_dict['best_f1']
best_acc = checkpoint['best_acc']
best_epoch= checkpoint['epoch']-1
print(f'best_epoch:{best_epoch}, best_acc:{best_acc}')
model.load_state_dict(checkpoint['state_dict'])

criterion = nn.CrossEntropyLoss()
# criterion = DynaLoss()
train_trans = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.2005,0.1490,0.1486],#均值标准差
                                                        std=[0.1445,0.1511,0.0967])])
infer_trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.2005,0.1490,0.1486],
                                                        std=[0.1445,0.1511,0.0967])])
test_dset = XDataset('data/test-%s.lib'%size, train_trans=train_trans, infer_trans=infer_trans)
test_loader = torch.utils.data.DataLoader(
    test_dset,
    batch_size=128,shuffle=False,
    pin_memory=True)

# len(test_dset.slideIDX)
# len(test_dset.targets)
# len(test_dset.slides)
test_dset.setmode(1)

# infos 就是要绘制的特征图
loss, probs, infos = inference_info(best_epoch, test_loader, model, criterion, 0)
torch.save(probs, 'data/probs.pth')
maxs = group_max(np.array(test_dset.slideIDX), probs, len(test_dset.targets))
tops = group_argtopk(np.array(test_dset.slideIDX), probs, k=1)
tops = [int(x) for x in tops]

info_name = 'info33'

info_dir = 'mil_test_info'
np.save("{}/{}.npy".format(info_dir, info_name), infos[tops])
np.save("{}/{}_label.npy".format(info_dir, info_name), test_dset.targets)
pred = [1 if x >= 0.5 else 0 for x in maxs]
tp, tn, fp, fn = tfpn(pred, test_dset.targets)
err = calc_err(pred, test_dset.targets)
acc = 1-err 
S, f1 = score(tp, tn, fp, fn)

print(f'acc:{acc}')