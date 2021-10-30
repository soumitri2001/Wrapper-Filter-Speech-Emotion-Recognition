import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SER_Network(nn.Module):
    def __init__(self, num_classes):
        super(SER_Network, self).__init__()
        self.backbone_ = torchvision.models.wide_resnet50_2(pretrained=True) # pre-trained backbone CNN 
        self.MODEL_NAME = "WIDE_RESNET50"
        self.base_model = nn.Sequential(*list(self.backbone_.children())[:-1]) # backbone CNN excluding last FC layer
        self.linear1 = nn.Linear(in_features=2048, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=num_classes)
        self.activation = nn.ReLU()
        
    
    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        lin = self.linear1(x)
        x = self.activation(lin)
        x = self.linear2(x)
        return lin, x


### utility function to train model ###
def train_model(args, model, criterion, optimizer, data_loader, phases):

    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    num_epochs = args.num_epochs

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        for phase in phases:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            epoch_loss = 0
            epoch_corrects = 0
            
            for ii, (images, labels, paths) in enumerate(data_loader[phase]):

                images = images.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'training'):
                    _, outputs = model(images)        
                    _,preds = torch.max(outputs,1) 
                    loss = criterion(outputs,labels)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                epoch_corrects += torch.sum(preds == labels.data)
                epoch_loss += loss.item() * images.size(0)

            epoch_accuracy = epoch_corrects/len(data_loader[phase]) 
            epoch_loss /= len(data_loader[phase])

            # store statistics
            if phase == 'training':
              epoch_loss = epoch_loss / args.batch_size
              train_loss.append(epoch_loss)
              epoch_accuracy = epoch_accuracy / args.batch_size
              train_acc.append(epoch_accuracy)
            if phase == 'validation':
              val_loss.append(epoch_loss)
              val_acc.append(epoch_accuracy)

            print(f'Epoch: [{epoch+1}/{num_epochs}] | Phase: {phase} | Loss: {epoch_loss:.6f} | Accuracy: {epoch_accuracy:.6f}')

            # deep copy the best model weights
            if phase == 'validation' and epoch_accuracy >= best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'====> Best accuracy reached so far at Epoch {epoch+1} | Accuracy = {best_accuracy:.6f}')
            torch.save(model.state_dict(), os.path.join(args.saved_models, f'{model.MODEL_NAME}.pth'))
        print('-------------------------------------------------------------------------')

    # training complete
    print(f'Best Validation Accuracy: {best_accuracy:4f}')
    model.load_state_dict(best_model_wts)

    history = {
        'train_loss' : train_loss.copy(),
        'train_acc' : train_acc.copy(),
        'val_loss' : val_loss.copy(),
        'val_acc' : val_acc.copy()
    }

    return model, history


### utility function to extract features ###
def extract_features(features, true_labels, paths, model, dataloader, phase):

    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        model.eval()
        for images, labels, p in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            true_labels.append(labels)
            paths.append(p)
            
            ftrs, outputs = model(images)
            _, preds = torch.max(outputs, 1)

            features.append(ftrs)
            n_samples += labels.size(0)
            n_correct += (preds == labels).sum().item()

        accuracy = n_correct/float(n_samples)
        print(f'Accuracy of model on {phase} set = {(100.0 * accuracy):.4f} %')

    return features, true_labels, paths


### get feature vectors from tensors ###
def get_features(features, true_labels, paths):
    ftrs = features.copy()
    lbls = true_labels.copy()

    for i in range(len(ftrs)):
        ftrs[i] = ftrs[i].cpu().numpy()
        lbls[i] = lbls[i].cpu().numpy()
        
    # convert to numpy array
    ftrs = np.array(ftrs)
    lbls = np.array(lbls)
    pths = np.array(paths)

    n_samples = ftrs.shape[0] * ftrs.shape[1]
    n_features = ftrs.shape[2]
    ftrs = ftrs.reshape(n_samples, n_features)

    n_lbls = lbls.shape[0]
    lbls = lbls.reshape(n_lbls)
    pths = pths.reshape(n_lbls)
    for i in range(pths.shape[0]):
        pths[i] = os.path.basename(pths[i])

    return ftrs, lbls, pths


### Testing model ###
if __name__ == "__main__":
    model = SER_Network(num_classes=4)
    model.to(device)
    vec = torch.ones((4,3,224,224), dtype=torch.float32)
    vec = vec.to(device)
    feat, out = model(vec)
    print(feat.shape, out.shape)
