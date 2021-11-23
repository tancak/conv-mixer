def accuracy(model, test_loader, loss_function, device):
    test_loss = []
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = loss_function(outputs,labels)
        test_loss.append(loss.item())
        
    return sum(test_loss)/len(test_loss)