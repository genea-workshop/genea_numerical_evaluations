import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from embedding_net import EmbeddingNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train_iter(target_data, net, optim):
    # zero gradients
    optim.zero_grad()

    # reconstruction loss
    feat, recon_data = net(target_data)
    recon_loss = F.l1_loss(recon_data, target_data, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))

    if True:  # use pose diff
        target_diff = target_data[:, 1:] - target_data[:, :-1]
        recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
        recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

    recon_loss = torch.sum(recon_loss)

    recon_loss.backward()
    optim.step()

    ret_dict = {'loss': recon_loss.item()}
    return ret_dict


def make_tensor(path, n_frames, stride=None):
    mean_vec = np.array([  0.4546838,  21.7154182,  -0.9253521,   1.0481254,  30.1385837,
                           0.3579833,   1.7905287,  38.52385  ,   1.6368418,   2.6107317,
                           49.0982409,   4.4734471,   3.1735519,  55.6535274,   6.7887565,
                           3.7772244,  62.2592191,   8.9336633,   2.1826697,  45.3888185,
                           7.9697471, -15.1223696,  47.1955581,   2.9793349, -27.7766248,
                           24.2758849,   3.1701817, -26.72694  ,  23.5385736,  20.4576112,
                           2.2024159,  45.3871329,   7.9702768,  19.7637237,  44.6687317,
                           4.375252 ,  27.5912466,  19.1198408,   5.9903543,  21.8780328,
                           16.8195207,  22.9530664])
    std_vec = np.array([  0.8220535,  0.1115134,  0.7292415,  1.6780658,  0.2834227,
                          1.4491553,  2.6036282,  0.5033388,  2.3467814,  3.8314201,
                          0.9260952,  3.5850173,  4.665101 ,  1.214391 ,  4.1563822,
                          5.5331544,  1.4715467,  4.6826288,  3.4655923,  1.2627345,
                          3.1362282,  3.4534337,  3.2095834,  3.7630859,  5.2367976,
                          9.1622181,  9.4337443, 10.0051307, 18.7604784, 12.0112832,
                          3.4653989,  1.2635428,  3.1361777,  3.3527777,  4.3935712,
                          3.7647275,  4.9868833,  7.7716174,  9.3219174, 10.0696823,
                          17.331519 , 11.2074617])

    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*.npz'))
    else:
        files = [path]

    samples = []
    stride = n_frames // 2 if stride is None else stride
    for file in files:
        data = np.load(file)
        data = data['clips']
        for i in range(0, len(data) - n_frames, stride):
            sample = data[i:i+n_frames]
            sample = (sample - mean_vec) / std_vec
            samples.append(sample)

    return torch.Tensor(samples)


def main(n_frames):
    # dataset
    train_dataset = TensorDataset(make_tensor('data/Trinity', n_frames))
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)

    # train
    loss_meters = [AverageMeter('loss')]

    # interval params
    print_interval = int(len(train_loader) / 5)

    # init model and optimizer
    pose_dim = 42
    generator = EmbeddingNet(pose_dim, n_frames).to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))

    # training
    for epoch in range(10):
        for iter_idx, target in enumerate(train_loader, 0):
            target = target[0]
            batch_size = target.size(0)
            target_vec = target.to(device)
            loss = train_iter(target_vec, generator, gen_optimizer)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | '.format(epoch, iter_idx + 1)
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                print(print_summary)

    # save model
    gen_state_dict = generator.state_dict()
    save_name = f'output/model_checkpoint_{n_frames}.bin'
    torch.save({'pose_dim': pose_dim, 'n_frames': n_frames, 'gen_dict': gen_state_dict}, save_name)


if __name__ == '__main__':
    n_frames = 90  # 30, 60, 90
    main(n_frames)
