import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from ConvMixer import ConvMixer
from utils import accuracy, validate
import wandb

import matplotlib.pyplot as plt
import random

TRAIN_ID = random.randint(0, 99999)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 80
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
NUM_CLASSES =100
EPOCHS = 25
DEPTH = 20
DIM = 512
KERNEL_SIZE = 7
PATCH_SIZE = 1

def train(net, loss_function, optimizer, train_loader, test_loader, epochs):
    running_loss = 0.0
    last_loss = 0.0
    
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

        wandb.log("Train Loss", last_loss)
        wandb.log("Test Loss", test_loss)
        wandb.log("Train Accuracy", train_acc)
        wandb.log("Test Accuracy", test_acc)
        wandb.watch(net)
        

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
    
    net = ConvMixer(num_classes = 100, dim = DIM, depth = DEPTH, kernel_size = KERNEL_SIZE, patch_size = PATCH_SIZE).to(DEVICE)

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
    
    wandb.init(project="conv-mixer", entity="tancak")
    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "depth": DEPTH,
        "dim": DIM,
        "kernel_size": KERNEL_SIZE,
        "patch_size": PATCH_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    }
    train(net, loss_function, optimizer, train_loader, test_loader, EPOCHS)
