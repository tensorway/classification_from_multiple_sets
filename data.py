# %%
from torchvision import transforms
from torchvision import transforms as T 
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, SVHN


easy_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

hard_transform = transforms.Compose([
    transforms.Resize(48),
    transforms.RandomCrop(32),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomSolarize(threshold=200, p=0.3),
    # transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

val_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])



def get_datasets (datasets, use_hard_transform=True) -> None:
    # ensure that datasets behaves like a set, reproducability
    datasets.sort()
    datasetsmap = {'cifar':CIFAR10, 'mnist':ColorMNIST, 'svhn':SVHNgoodApi}
    nclassesmap = {'cifar':10, 'mnist':10, 'svhn':10}
    if use_hard_transform:
        train_transfrom = hard_transform
    else:
        train_transfrom = easy_transform
    trainset = [
        datasetsmap[s](root='./data', train=True, download=True, transform=train_transfrom) 
        for s in datasets
    ]
    valset = [
        datasetsmap[s](root='./data', train=False, download=True, transform=val_transform) 
        for s in datasets
    ]
    nclasses = [ nclassesmap[d] for d in datasets ]
    return MultiDataset(trainset), MultiDataset(valset), sum(nclasses)




class MultiDataset(Dataset):
    def __init__(self, datasets) -> None:
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, index):
        cumnum = 0
        addclasses = 0
        for dataset in self.datasets:
            if cumnum <= index < cumnum + len(dataset):
                img, label = dataset[index-cumnum]
                return img, label+addclasses
            cumnum += len(dataset)
            addclasses += len(dataset.classes)

        raise IndexError()

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])    

    def get_classes(self):
        l = []
        for dataset in self.datasets:
            l += dataset.classes
        return l


class ColorMNIST(Dataset):
    def __init__(self, root, train, download, transform) -> None:
        self.mnist = MNIST(root=root, train=train, download=download)
        self.transform = transform
        self.classes = [str(i)+'-mnist' for i in range(10)]

    def __getitem__(self, index):
        img, label = self.mnist[index]
        img = self.transform(img.convert('RGB'))
        
        return img, label
    def __len__(self):
        return len(self.mnist)


class SVHNgoodApi(SVHN):
    def __init__(self, root, train, download, transform) -> None:
        split = 'train' if train else 'test'
        super().__init__(root=root, transform=transform, download=download, split=split)
        self.transform = transform
        self.classes = [str(i)+'-svhn' for i in range(10)]

    def __getitem__(self, index):
        return super().__getitem__(index)
        
    def __len__(self):
        return super().__len__()



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t, v, c = get_datasets(['cifar', 'mnist', 'svhn'], True)
    img = v[10][0];
    plt.imshow(img.permute(1, 2, 0).numpy());
    v[0][1]

    v[100_000][1]


# %%
