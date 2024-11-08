import argparse
import torch
import numpy as np
from tqdm import tqdm
from load import load_data
from vision_module import vision_module
from create_datasets import ShapesDataset

def get_params():

    parser = argparse.ArgumentParser(prog="Train Visual Model", description="trains a vision module")

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for the training")
    
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help="Weight decay for the training")
    
    parser.add_argument('--epochs', type=int, default=15,
                        help="How many epochs to train for")
    
    parser.add_argument('--test_every', type=int, default=5, 
                        help="After how many training epochs each test runs are to be conducted")
    
    parser.add_argument('--save_model', type=bool, default=True,
                        help="Use if you want to save the model after training")
    
    parser.add_argument('--train_split', type=float, default=0.8,
                        help="Determine how much of the dataset will be used for training and how much for testing")
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Give batch size for training")
    
    args = parser.parse_args()

    return args

def transform_data(data):
    data = data.astype("float32") / 255.0
    # Compute the mean per channel across all images and pixels
    mean = np.mean(data, axis=(1, 2), keepdims=True)
    data = data - mean
    return data

def train(args):
    # first load the dataset and define all parts necessary for the training
    # try to load the dataset if it was saved before
    dataset_path = ('dataset/complete_dataset_20241027_norm')
    try:
        complete_data = torch.load(dataset_path)
        print('Dataset was found and loaded successfully')

    # otherwise create the dataset and save it to the folder for repeated use
    except:
        print('Dataset was not found, creating it instead...')
        full_data, labels_reg, _ = load_data(normalize=False, subtract_mean=False)
        complete_data = ShapesDataset(full_data, labels_reg, transform=None)
        torch.save(complete_data, dataset_path)
        print(f'Dataset was created and saved to {dataset_path}')

    complete_data = ShapesDataset(complete_data.images, complete_data.labels, transform=transform_data)
    train_size = int(args.train_split * len(complete_data))
    test_size = len(complete_data) - train_size

    print(f'Training on {train_size} samples and testing on {test_size} samples')
    
    train_data, test_data = torch.utils.data.random_split(complete_data, [train_size, test_size])

    # init both dataloaders with corresponding datasets
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True)

    # init vision module to train
    model = vision_module(num_classes=64)

    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use NVIDIA GPU
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')  # Use Apple GPU
    else:
        device = torch.device('cpu')

    model.to(device)

    # use SGD as optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr = args.learning_rate,
                                weight_decay = args.weight_decay)
    
    # use crossentropyloss as loss
    criterion = torch.nn.CrossEntropyLoss()

    # now after everything was set up the actual training starts
    # save losses and accuracies as lists for later printing
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # train for given nr of epochs
    print('Starting training for ', args.epochs, ' epochs\n')
    for epoch in range(args.epochs):
        print("Epoch ", epoch+1)
        model.train()
        train_loss, train_acc = 0., 0.
        
        for i, (input, target) in tqdm(enumerate(train_loader)):
            input = input.to(device)
            target = target.to(device=device, dtype=torch.float32)
            
            # compute network prediction and use it for loss
            output = model(input)
            output = output.to(device=device)
            loss = criterion(output, target)
            train_loss += loss.item()
            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracies = torch.argmax(output, dim=1) == torch.argmax(target, dim=1)
            batch_accuracy = torch.mean(accuracies.float())
            train_acc += batch_accuracy.item()

        # add accuracy and loss for each epoch
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = train_acc / len(train_loader)
        train_accuracies.append(epoch_acc)
        train_losses.append(epoch_loss)

        # print most recent entry to our loss and accuracy lists
        print("Epoch ", epoch+1, " training loss: ~", round(train_losses[-1], 5)," training accuracy: ~", round(train_accuracies[-1], 5), "\n")

        # test every X epoch as given by test_every
        if ((epoch+1) % args.test_every) == 0:
            test_loss, test_acc = test(test_loader, model, criterion)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
    
    if args.save_model:
        model_path = 'models/vision_module_mps'
        model.cpu()
        torch.save(model.state_dict(), model_path)
        print(f'Model was saved to {model_path}')
    
    return train_losses, train_accuracies, test_losses, test_accuracies


def test(test_loader, model, criterion):    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    total_loss, total_accuracy = 0., 0.

    with torch.no_grad():
         print('Starting test run')
         for i, (input, target) in tqdm(enumerate(test_loader)):
            input = input.to(device)
            target = target.to(device=device, dtype=torch.float32)
            
            # compute network prediction and use it for loss
            output = model(input)
            output = output.to(device=device)
            loss = criterion(output, target)
            total_loss += loss.item()
            accuracies = torch.argmax(output, dim=1) == torch.argmax(target, dim=1)
            total_accuracy += torch.mean(accuracies.float()).item()

    # add accuracy and loss
    epoch_loss = total_loss / len(test_loader)
    epoch_acc = total_accuracy / len(test_loader)

    print("test loss was: ~", round(epoch_loss, 5), "and test accuracy was: ~", round(epoch_acc, 5), "\n")
    
    return epoch_loss, epoch_acc

if __name__ == "__main__":

    args = get_params()

    train_losses, train_accuracies, test_losses, test_accuracies = train(args)

    # round losses and accuracies for prettier printing
    rounded_train_losses        = [round(loss, 3) for loss in train_losses]
    rounded_train_accuracies    = [round(acc, 3) for acc in train_accuracies]
    rounded_test_losses         = [round(loss, 3) for loss in test_losses]
    rounded_test_accuracies     = [round(acc, 3) for acc in test_accuracies]

    print('training losses were: ~', rounded_train_losses, "\n")
    print('training accuracies were: ~', rounded_train_accuracies, "\n")

    print('test losses were: ~', rounded_test_losses, "\n")
    print('test accuracies were: ~', rounded_test_accuracies, "\n")