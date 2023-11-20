import argparse
import torch
print(torch.__version__)
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import time


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=10)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args


def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(128),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data



def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(128),
                                      transforms.CenterCrop(128),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=64)
    return loader



def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device


def primaryloader_model(architecture):
    
    args=arg_parser()
    
    if args.arch == "vgg13":
        model = models.vgg13(pretrained=True)
        model.name="vgg13"
        
    #default to vgg16 if vgg13 is not selected
    else :
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        
    for param in model.parameters():
        param.requires_grad = False 
    return model


def initial_classifier(model, hidden_units):
    
    from collections import OrderedDict

    classifier = nn.Sequential(OrderedDict([
        ('flatten', Flatten()),  # Flatten layer
        ('fc1', nn.Linear(8192, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(0.05)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    return classifier



def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for i, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy




def network_trainer(model, trainloader, validloader, Device, 
                  criterion, optimizer, Epochs, print_every, steps):
    
    if type(Epochs) == type(None):
        Epochs = 10
        print("10 epochs.")    
 
    print("Training started .....\n")

    # Train Model
    for e in range(Epochs):
        running_loss = 0
        model.train() 
        
        for i, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(Device), labels.to(Device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Running Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Running Training Accuracy: {:.2f}% |".format(running_accuracy / print_every * 100),
                      "Validation Loss: {:.4f} | ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))

                        
                
                torch.cuda.empty_cache()
            
                running_loss = 0
                model.train()

    return model



#Validate model
def validate_model(model, testloader, Device):
   # Do validation on the test set
    correct,total = 0,0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy: %d%%' % (100 * correct / total))
    

# Function initial_checkpoint(Model, Save_Dir, Train_data) saves the model at a defined checkpoint
def initial_checkpoint(Model, Save_Dir, Train_data):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            model.class_to_idx = image_datasets['train'].class_to_idx
            torch.save({'structure' :'alexnet',
                        'hidden_layer1':120,
                         'droupout':0.5,
                         'epochs':12,
                         'state_dict':model.state_dict(),
                         'class_to_idx':model.class_to_idx,
                         'optimizer_dict':optimizer.state_dict()},
                         'checkpoint.pth')
            model.class_to_idx = Train_data.class_to_idx

                # Create checkpoint dictionary
            checkpoint = {'architecture': model.name,
                          'classifier': model.classifier,
                          'class_to_idx': model.class_to_idx,
                          'state_dict': model.state_dict()}

                # Save checkpoint
            torch.save(checkpoint, 'my_checkpoint.pth')
        else: 
                print("Directory not found, model will not be saved.")

def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = test_transformer(train_dir)
    valid_data = train_transformer(valid_dir)
    test_data = train_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)

    model = primaryloader_model(architecture=args.arch)
    
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    
    device = check_gpu(gpu_arg=args.gpu);
    model.to(device);
    
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 20
    steps = 0
    
    trained_model = network_trainer(model, trainloader, validloader,device, criterion, optimizer, args.epochs, print_every, steps)
    
    print("\nTraining process is completed!!")
    
    validate_model(trained_model, testloader, device)
   
    initial_checkpoint(trained_model, args.save_dir, train_data)
if __name__ == '__main__': main()
