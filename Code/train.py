import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from ConvMixer import ConvMixer
from utils import accuracy, validate
from koila import lazy

import matplotlib.pyplot as plt
import random

TRAIN_ID = random.randint(0, 99999)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 90
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
NUM_CLASSES=100
EPOCHS=500

def train(net, loss_function, optimizer, train_loader, test_loader, epochs):
    tb = SummaryWriter()
    running_loss = 0.0
    last_loss = 0.0
    
    for epoch in range(epochs):
        net.train()
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = lazy(data[0]).to(DEVICE), lazy(data[1]).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if i % 4 == 3:
                print('Epoch {0} [{1}/{2}]: Training Loss: {3:0.3f}'.format(epoch + 1,
                        (i + 1) * BATCH_SIZE, len(train_loader) * BATCH_SIZE,
                        running_loss / 4), end='\r')
                last_loss = running_loss / 4
                running_loss = 0.0
                
        net.eval()
        test_loss = validate(net, test_loader, loss_function, DEVICE)
        test_acc = accuracy(net, test_loader, DEVICE)
        train_acc = 0#accuracy(net, train_loader, DEVICE)

        tb.add_scalar("Train Loss", last_loss, epoch)
        tb.add_scalar("Test Loss", test_loss, epoch)
        tb.add_scalar("Train Accuracy", train_acc, epoch)
        tb.add_scalar("Test Accuracy", test_acc, epoch)
        

        print('Epoch {0}\033[K\nTraining Loss: {1:0.3f}          Test Loss: {2:0.3f}          Training Accuracy:{3:0.3f}          Test Accuracy:{4:0.3f}'.format(epoch + 1, last_loss, test_loss, train_acc, test_acc))
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': last_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            }, str(TRAIN_ID)+ "-" + str(epoch) +".pkl")
    tb.close()

if __name__ == '__main__':
    print("Params:")
    print(("  Training ID: {0:05d}\n"
           "  Device: {1}\n"
           "  Batch Size: {2}\n"
           "  Learning Rate: {3}\n"
           "  Weight Decay: {4}\n"
           "  Max Epochs: {5}\n").format(TRAIN_ID, DEVICE, BATCH_SIZE,
                                     LEARNING_RATE, WEIGHT_DECAY, EPOCHS))
    print("Generating model and optimizer")
    
    net = ConvMixer(num_classes = 100, dim = 512, depth = 20, kernel_size = 7, patch_size = 1).to(DEVICE)

    if DEVICE == 'cuda':
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, 
                            weight_decay=WEIGHT_DECAY)
    
    print("Making datasets and dataloaders")
                                                  
    transform_train = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.RandomRotation(1),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, 
                                 transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True, num_workers=2)
    
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, 
                                transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    
    print("Starting training")
    
    train(net, loss_function, optimizer, train_loader, test_loader, EPOCHS)
