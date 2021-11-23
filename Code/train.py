import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from ConvMixer import ConvMixer
from utils import accuracy

import matplotlib.pyplot as plt
import random

TRAIN_ID = random.randint(0, 99999)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NUM_CLASSES=10
EPOCHS=25
LOG_DIR='./logs/train-log-{0:05d}.csv'.format(TRAIN_ID)

def train(net, loss_function, optimizer, train_loader, test_loader, epochs):
    running_loss = 0.0
    
    for epoch in range(epochs):
        net.train()
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if i % 100 == 99:
                print('Epoch {0}: [{1}/{2}] Training Loss: {3:0.3f}'.format(epoch + 1,
                        (i + 1) * BATCH_SIZE, len(train_loader) * BATCH_SIZE,
                        running_loss / 100), end='\r', flush=True)
                running_loss = 0.0
                
        net.eval()
        test_loss = accuracy(net, test_loader, loss_function, DEVICE)
        print('Epoch {0} \t\t Training Loss: {1} \t\t Test Loss:{2}'.format(epoch + 1, running_loss / 100, test_loss))
        

if __name__ == '__main__':
    print("Params:")
    print(("  Training ID: {0:05d}\n"
           "  Device: {1}\n"
           "  Batch Size: {2}\n"
           "  Learning Rate: {3}\n"
           "  Weight Decay: {4}\n"
           "  Max Epochs: {5}\n"
           "  Log Path: {6}").format(TRAIN_ID, DEVICE, BATCH_SIZE,
                                     LEARNING_RATE, WEIGHT_DECAY, EPOCHS, 
                                     LOG_DIR))
    print("Generating model and optimizer")
    
    net = ConvMixer(10).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, 
                            weight_decay=WEIGHT_DECAY)
    
    print("Making datasets and dataloaders")
                                                  
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, 
                                 transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True, num_workers=2)
    
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, 
                                transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    
    print("Starting training")
    
    train(net, loss_function, optimizer, train_loader, test_loader, EPOCHS)