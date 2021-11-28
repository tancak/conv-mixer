import torch

def validate(model, test_loader, loss_function, device):
    test_loss = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs,labels)
            test_loss.append(loss.item())
        
    return sum(test_loss)/len(test_loss)

def accuracy(model, dataloader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    return correct / total