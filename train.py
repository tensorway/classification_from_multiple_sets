#%%
import time
import torch
import torchvision
import torch as th
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from models import CNN
from utils import get_confusion_matrix_img, load_model, save_model, seed_everything
from data import get_datasets
import argparse
from torch.utils.tensorboard import SummaryWriter
import logging



# Argument parsing #################################################################################
def add_standard_arguments(parser):
    parser.add_argument('-d','--datasets', type=str, nargs='+', help='set of datasets to use, options: cifar, mnist, svhn', default=['mnist'])
    parser.add_argument('-p','--projector_shape', type=int, nargs='+', help='projector shape', default=[512, 512, 512])
    parser.add_argument("-s", "--seed", type=int, default=42, help="RNG seed. Default: 42.")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="train and valid batch size")
    parser.add_argument("-e", "--n_total_epochs", type=int, default=150, help="total number of epochs")
    parser.add_argument("-lr", "--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("-sda", "--save_delta_all", type=int, default=1500, help="in seconds, the model that is stored and overwritten to save space")
    parser.add_argument("-sdr", "--save_delta_revert", type=int, default=3000, help="in seconds, checkpoint models saved rarely to save storage")
    parser.add_argument("-chp", "--checkpoints_path", type=str, default='model_checkpoints/', help="folder where to save the checkpoints")
    parser.add_argument("-ptr", "--pretrained_model_path", type=str, help="pretrained model path from which to train")
    parser.add_argument("-fin", "--final_model_path", type=str, help="final model path where to save the model", default=None)
    parser.add_argument("-r", "--results_path", type=str, default='results.txt', help="file where to save the results")







# Main loop #################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification from multiple sets script.')
    add_standard_arguments(parser)
    args = parser.parse_args()
    logging.info('args are:')
    logging.info(args)

    seed_everything(args.seed)

    trainset, valset, n_classes = get_datasets(args.datasets, use_hard_transform=False)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_dataloader = torch.utils.data.DataLoader(valset,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    model_str = str(args.datasets) # args.__str__()
    model_path = Path(args.checkpoints_path)/('model_'+model_str+'.pt')
    optimizer_path =  Path(args.checkpoints_path)/('optimizer_'+model_str+'.pt')

    model = CNN(args.projector_shape, n_classes=n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info("using", device)
    load_model(model, args.pretrained_model_path)

    opt = th.optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter('tensorboard/finetune_'+model_str.split('/')[-1])

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
        writer.add_scalar("Loss_finetune/valid", loss, step)
        writer.add_scalar("acc/valid", acc, step)
        writer.add_image("confusion", get_confusion_matrix_img(confusion_mat.cpu().numpy(), valset.get_classes()), step, dataformats='HWC')
        model.train()
        return acc


    # Main loop #################################################################################

    step = 0
    t_last_save_revert = time.time()
    t_last_save_all = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for ep in range(args.n_total_epochs):
        model.backbone.eval()
        for ibatch, (imgs, labels) in enumerate(train_dataloader):

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                imgs, labels = imgs.to(device), labels.to(device) 
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                acc = (outputs.argmax(dim=-1) == labels).float().mean().item()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if ibatch%30 == 0:
                writer.add_scalar("Loss_finetune/train", loss, step)
                writer.add_scalar("acc/train", acc, step)
                print(ep, step, loss.item(), acc)
            if ibatch % 300 == 0:
                whole_dataset_eval()

            if time.time() - t_last_save_all > args.save_delta_all:
                save_model(model, str(model_path))
                save_model(opt, str(optimizer_path))
                t_last_save_all = time.time()

            if time.time() - t_last_save_revert > args.save_delta_revert:
                save_model(model, str(model_path).split('.pt')[0] + str(step) + '.pt')
                save_model(opt, str(optimizer_path).split('.pt')[0] + str(step) + '.pt')
                t_last_save_revert = time.time()
            
            step += 1

            

    # Write results #################################################################################
    with open(args.results_path, 'a') as fout:
        fout.write('acc='+str(acc))
        dic = vars(args)
        for k in sorted(dic.keys()):
            fout.write(','+str(k)+'='+str(dic[k]))

    if args.final_model_path:
        save_model(model, args.final_model_path)
