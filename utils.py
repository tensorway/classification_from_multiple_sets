# %%
import io
import torch as th
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def load_model(model, path):
    try:
        model.load_state_dict(th.load(path))
        print(f"loaded model ({type(model).__name__}) from {path}")
    except:
        print(f"could not load model ({type(model).__name__}) from {path}")

def save_model(model, path):
    th.save(model.state_dict(), path)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def tensor_img_show(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def tensor_mat_show(img):
    npimg = img.detach().cpu().numpy()
    plt.imshow(npimg)
    plt.show()


def get_confusion_matrix_img(confusion_matrix, classes, img_scale = 1.5):
    confusion_matrix = confusion_matrix.astype(np.float32).tolist()
    fontsize = int(img_scale*20)
    matplotlib.rcParams['figure.figsize'] = [
        int(len(classes)*img_scale), int(len(classes)*img_scale)
    ]
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(classes)), labels=classes)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, f'{confusion_matrix[i][j] : 4.2f}',
                        ha="center", va="center", color="g", fontweight=500, fontsize=fontsize)

    ax.set_title("Confusion matrix", fontsize=int(fontsize*2.5))
    plt.xlabel('ground truth', fontsize=int(fontsize*1.5))
    plt.ylabel('predictions', fontsize=int(fontsize*1.5))
    fig.tight_layout()
    # plt.show()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1)).astype(np.uint8)
    plt.imshow(im)
    return im





# %%
if __name__ == '__main__':
    conf = [
        [0.1, 0.9],
        [0.2, 0.8],
    ]
    get_confusion_matrix_img(conf, ['a', 'b'])
# %%
