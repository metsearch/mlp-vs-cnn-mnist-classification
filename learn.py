import os
import torch as th
from torch.autograd import Variable

from rich.progress import track

from arch import MLP_Model, CNN_Model
from utilities.utils import *

def learn_with_mlp(data_loader, nb_epochs: int, device: th.device, path2model: str, path2metrics: str):
    layer_cfg=[784, 512, 256, 64, 10]
    net = MLP_Model(
        layer_cfg=layer_cfg,
        non_linears=[1, 1, 1, 0],
        dropouts=[0.1, 0.1, 0.1, 0.0]
    )
    print(net)
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    criterion = th.nn.CrossEntropyLoss()
    
    losses = []
    nb_data = len(data_loader)
    for epoch in range(nb_epochs):
        counter = 0
        epoch_loss = 0.0
        
        for X, Y in track(data_loader, description=f'Training...'):
            X = Variable(X)
            Y = Variable(Y)
            X = X.view(-1, layer_cfg[0])

            X = X.to(device)
            Y = Y.to(device) 

            P = net(X)
            optimizer.zero_grad()
            E: th.Tensor = criterion(P, Y)
            E.backward()
            optimizer.step()
            
            epoch_loss += E.cpu().item()
            counter += len(X)
            
        average_loss = epoch_loss / nb_data
        losses.append(average_loss)
        logger.debug(f'[{epoch:03d}/{nb_epochs:03d}] [{counter:05d}/{nb_data:05d}] >> Loss : {average_loss:07.3f}')

    th.save(net.cpu(), os.path.join(path2model, 'mlp_network.th'))
    logger.info('The model was saved ...!')
    
    plt.plot(range(1, nb_epochs + 1), losses, label='MLP Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(path2metrics, 'mlp_training_loss.png'))
    plt.show()
    
def learn_with_cnn(data_loader, nb_epochs: int, device: th.device, path2model: str, path2metrics: str):
    layer_cfg = {
        'convs': {
            'shapes': [1, 16, 32, 64],
            'dropouts': [0.2, 0.2, 0.2]
        },
        'linears': {
            'shapes': [64*7*7, 64, 10],
            'non_linears': [1, 0],
            'dropouts': [0.2, 0.0]
        }
    }
    net = CNN_Model(layer_cfg)
    print(net)
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    criterion = th.nn.CrossEntropyLoss()
    
    losses = []
    nb_data = len(data_loader)
    for epoch in range(nb_epochs):
        counter = 0
        epoch_loss = 0.0
        
        for X, Y in track(data_loader, description=f'Training...'):
            X = Variable(X)
            Y = Variable(Y)

            X = X.to(device)
            Y = Y.to(device) 

            P = net(X)
            optimizer.zero_grad()
            E: th.Tensor = criterion(P, Y)
            E.backward()
            optimizer.step()
            
            epoch_loss += E.cpu().item()
            counter += len(X)
            
        average_loss = epoch_loss / nb_data
        losses.append(average_loss)
        logger.debug(f'[{epoch:03d}/{nb_epochs:03d}] [{counter:05d}/{nb_data:05d}] >> Loss : {average_loss:07.3f}')

    th.save(net.cpu(), os.path.join(path2model, 'cnn_network.th'))
    logger.info('The model was saved ...!')
    
    plt.plot(range(1, nb_epochs + 1), losses, label='CNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(path2metrics, 'cnn_training_loss.png'))
    plt.show()

if __name__ == '__main__':
    logger.info('Learning test...')