import argparse
import os
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils import data
from utils import data_loaders, image_folder
parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')
parser.add_argument('data_dir', type=str, help='Directory of training data')
parser.add_argument('--save_dir', type=str, default='./', help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=1024, help='Number of hidden units')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args()
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'

if not os.path.exists(train_dir):
    print("Training set directory doesn't exist: {}".format(train_dir))
    raise FileNotFoundError
if not os.path.exists(valid_dir):
    print("Validation set directory doesn't exist: {}".format(valid_dir))
    raise FileNotFoundError
    
if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
elif args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
else:
    print("Not implemented")
    raise NotImplementedError

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1))

model.classifier = classifier

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
model.to(device)

print('training..')
epochs = args.epochs
for epoch in range(epochs):
    print(f'Epoch number {epoch}')
    model.train()
    running_loss = 0

    for images, labels in data_loaders('train', train_dir):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    model.eval()
    valid_loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in data_loaders('val', valid_dir):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            valid_loss += criterion(outputs, labels)

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {running_loss/len(data_loaders('train', train_dir)):.3f}.. "
          f"Validation loss: {valid_loss/len(data_loaders('val', valid_dir)):.3f}.. "
          f"Validation accuracy: {accuracy/len(data_loaders('val', valid_dir)):.3f}")
    
model.class_to_idx = image_folder('train', train_dir).class_to_idx
print('saving...')
# checkpoint dictionary
checkpoint = {'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'optimizer_state': optimizer.state_dict(),
              'epochs': epochs}
save_dir = args.save_dir + '/checkpoint.pth'    
# save
torch.save(checkpoint, save_dir)