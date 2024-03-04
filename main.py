import click

import torchvision as tv
import torch as th

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader

from learn import *
from inference import *

from utilities.utils import *

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='debug mode flag', default=False)
@click.pass_context
def router_cmd(ctx: click.Context, debug):
    ctx.obj['debug_mode'] = debug 
    invoked_subcommand = ctx.invoked_subcommand 
    if invoked_subcommand is None:
        logger.debug('no subcommand were called')
    else:
        logger.debug(f'{invoked_subcommand} was called')
        
@router_cmd.command()
# @click.option('--path2data', help='path to data', type=click.Path(False), default='data/train_data.pkl')
@click.option('--nb_epochs', help='number of epochs', type=int, default=10)
@click.option('--bt_size', help='batch size', type=int, default=32)
@click.option('--arch', help='cnn or mlp', type=click.Choice(['cnn', 'mlp']), default='mlp')
def learning(nb_epochs, bt_size, arch):
    logger.debug('Learning...')
    path2model = 'models/'
    if not os.path.exists(path2model):
        os.makedirs(path2model)
        
    path2metrics = 'metrics/'
    if not os.path.exists(path2metrics):
        os.makedirs(path2metrics)
        
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(
        MNIST('../mnist_data', 
            download=True, 
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
            ])), 
            batch_size=bt_size, 
            shuffle=True
    )
    
    if arch == 'mlp':
        learn_with_mlp(train_loader, nb_epochs, device, path2model, path2metrics)
    else:
        learn_with_cnn(train_loader, nb_epochs, device, path2model, path2metrics)
    

@router_cmd.command()
@click.option('--path2models', help='path to models', type=click.Path(True), default='models/')
@click.option('--model_name', help='vectorization', type=str, default='mlp_network.th')
@click.option('--bt_size', help='batch size', type=int, default=32)
@click.option('--arch', help='cnn or mlp', type=click.Choice(['cnn', 'mlp']), default='mlp')
def inference(path2models, model_name, bt_size, arch):
    logger.debug('Inference...')
    path2metrics = 'metrics/'
    if not os.path.exists(path2metrics):
        os.makedirs(path2metrics)
    model = th.load(os.path.join(path2models, model_name))
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_loader = DataLoader(
        MNIST('../mnist_data', 
            download=True, 
            train=False,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), 
        batch_size=bt_size,
        shuffle=True
    )
    
    model.eval()
    if arch == 'mlp':
        images, predictions, ground_truths = mlp_inference(test_loader, model, device)
    else:
        images, predictions, ground_truths = cnn_inference(test_loader, model, device)

    display_images_with_predictions(path2metrics, arch, images, predictions, ground_truths)
    plot_confusion_matrix(path2metrics, arch, ground_truths, predictions, list(range(10)))

if __name__ == '__main__':
    router_cmd(obj={})