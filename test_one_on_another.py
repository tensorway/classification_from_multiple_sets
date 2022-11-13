#%%
import torch
import torch as th
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from models import CNN
from utils import get_confusion_matrix_img, load_model, seed_everything
import matplotlib.pyplot as plt
from data import get_datasets
import argparse
from torch.utils.tensorboard import SummaryWriter
import logging



# Argument parsing #################################################################################
def add_standard_arguments(parser):
    parser.add_argument('-d','--dataset', type=str, help='dataset on which to evaluate', default='mnist')
    parser.add_argument('-p','--projector_shape', type=int, nargs='+', help='projector shape', default=[512, 512, 512])
    parser.add_argument("-ptr", "--pretrained_model_path", type=str, help="pretrained model path which to evaluate")
    parser.add_argument("-img", "--img_path", type=str, help="where to save confusion matrix")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="train and valid batch size")




# Main loop #################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification from multiple sets script.')
    add_standard_arguments(parser)
    args = parser.parse_args()
    logging.info('args are:')
    logging.info(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("using", device)


    trainset, valset, n_classes = get_datasets([args.dataset], use_hard_transform=False)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_dataloader = torch.utils.data.DataLoader(valset,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = CNN(args.projector_shape, n_classes=n_classes)
    model.to(device)
    
    load_model(model, args.pretrained_model_path)

    criterion = nn.CrossEntropyLoss()

    def whole_dataset_eval():
        model.eval()
        cumacc = 0
        cumloss = 0
        guesses = []
        for imgs, labels in valid_dataloader:
            with th.no_grad():
                with torch.cuda.amp.autocast():
                    imgs, labels = imgs.to(device), labels.to(device) 
                    outputs = model(imgs)
                    cumloss += criterion(outputs, labels)
                    cumacc += (outputs.argmax(dim=-1) == labels).float().mean().item()
                    guesses.append( outputs.argmax(dim=-1)*n_classes + labels)
        acc, loss = cumacc/len(valid_dataloader), cumloss/len(valid_dataloader)
        guesses = torch.cat(guesses)
        confusion_mat = torch.bincount(guesses, minlength=n_classes**2).view(n_classes, n_classes).float()
        confusion_mat = confusion_mat / confusion_mat.sum(dim=0, keepdim=True)
        confusion_mat = torch.nan_to_num(confusion_mat, 0)
        confusion_img = get_confusion_matrix_img(confusion_mat.cpu().numpy(), valset.get_classes())

        return acc, loss, confusion_img

    acc, loss, confusion_img = whole_dataset_eval()
    print('Accuracy:', acc)
    print('Loss:', loss.item())
    plt.imsave(args.img_path, confusion_img)


    

