#%%
import torch as th
import torch
from torch import nn
import torch.nn.functional as F

from data import get_datasets


class CNN(nn.Module):
    def __init__(
        self, 
        projector_mlp_arch = [128],
        n_classes = 10,
        temperature = 1,
        train_temperature = False
    ):
        '''
        Args:
            - projector_mlp_arch: List[int] 
                architecture of the mlp that will be appended to the backbone 
                does not include the backbone last layer
        '''
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
        )

        self.projector = MLP(net_arch=[2048]+projector_mlp_arch+[n_classes])
        if train_temperature:
            self.temperature = nn.parameter.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.projector(x)
        return th.softmax(x / self.temperature, dim=-1)

class MLP(nn.Module):
    '''
    accepts layer sizes and creates a MLP model out of it
    Args:
        net_arch: list of integers denoting layer sizes (input, hidden0, hidden1, ... hiddenn, out)
    '''
    def __init__(self, net_arch, last_activation= lambda x:x):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(b) for b in net_arch[1:-1]])
        self.last_activation = last_activation
    def forward(self, x):
        h = x
        for lay, norm in zip(self.layers[:-1], self.batch_norms):
            h = norm(F.relu(lay(h)))
        h = self.layers[-1](h)
        return self.last_activation(h)

if __name__ == '__main__':
    t, v, nclasses = get_datasets(['mnist'])
    model = CNN(n_classes=nclasses)
    model.eval()

    nump = 0
    for p in model.parameters():
        nump += p.numel()
    print('model has ', nump/10**6, 'M parameters')

    img = v[0][0]
    print(img.shape)
    model(img[None])


# %%
