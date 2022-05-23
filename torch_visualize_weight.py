import torch
import torchvision.models as models
from torchvision import utils
from matplotlib import pyplot as plt
import numpy as np

def visTensor(layername, tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape
    print(f'Tensor shape {tensor.shape}')

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    image = grid.detach().numpy().transpose((1, 2, 0))
    print(f'Name {layername}, Filter shape {image.shape}')
    new_data = np.zeros([image.shape[0] * 10, image.shape[1] * 10, image.shape[2]])

    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            new_data[j * 10: (j+1) * 10, k * 10: (k+1) * 10, :] = image[j, k, :]
    plt.imsave(layername + '.png', new_data)

def densityW(layername, tensor):
    tensor = tensor[:,:,0,0]
    tensor = tensor.detach().numpy()
    l1_norm = np.linalg.norm(tensor, ord=1, axis=1, keepdims=False)
    print(f'Tensor shape {tensor.shape}')
    print(f'L1 norm shape {l1_norm.shape}')
    fig = plt.hist(l1_norm, bins=100)
    plt.title(layername+'Filter L1 norm')
    plt.xlabel("l1_norm value")
    plt.ylabel("Density")
    plt.savefig(layername+".png")
    plt.close('all')

    # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # # We can set the number of bins with the `bins` kwarg
    # axs[0].hist(x, bins=n_bins)
    # axs[1].hist(y, bins=n_bins)


if __name__ == "__main__":
    from MobileNetV1_ImageNet import MobileNet
    model = MobileNet(num_classes=1000)
    # Load pre-trained model
    PATH = '../models/imagenet_mobilenet_full_model.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model_dict = model.state_dict()
    model_names = []
    model_params = []
    for name, param in model.named_parameters():
        # model_dict[name] = param
        model_names.append(name)
        model_params.append(param)
        # print(f'Name: {name}')
    
    layer1 = 'layers.9.conv2.weight'
    layer2 = 'layers.10.conv2.weight'
    layer3 = 'layers.11.conv2.weight'
    layer4 = 'layers.12.conv2.weight'
    densityW(layer1, model_params[model_names.index(layer1)])
    densityW(layer2, model_params[model_names.index(layer2)])
    densityW(layer3, model_params[model_names.index(layer3)])
    densityW(layer4, model_params[model_names.index(layer4)])

    # visTensor(layer1, model_params[model_names.index(layer1)], ch=0, allkernels=False)
    # visTensor(layer2, model_params[model_names.index(layer2)], ch=0, allkernels=False)
    # visTensor(layer3, model_params[model_names.index(layer3)], ch=0, allkernels=False)
    # visTensor(layer4, model_params[model_names.index(layer4)], ch=0, allkernels=False)

