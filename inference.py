import torch as th
from torch.autograd import Variable

from rich.progress import track

from utilities.utils import *

def mlp_inference(test_loader, model, device):
    predictions = []
    ground_truths = []
    images = []
    
    with th.no_grad():
        for X, Y in track(test_loader, description='Inference...'):
            X = Variable(X)
            Y = Variable(Y)
            X = X.view(-1, 28*28)
            
            X = X.to(device)
            Y = Y.to(device)

            P = model(X)
            predictions.extend(th.argmax(P, dim=1).cpu().numpy())
            ground_truths.extend(Y.cpu().numpy())
            images.extend(X.cpu().numpy())

    model.train()

    return images, predictions, ground_truths

def cnn_inference():
    pass

if __name__ == '__main__':
    logger.info('Inference test...')