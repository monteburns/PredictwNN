import argparse
import dataops as d
from model import initialize_model
from train import train_model
from util import feature_importance, convert_to_images, convert_to_images_raw, test_model
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt


def main(args):


    if args.create_data:
        data = d.Dataset('u')

        print("Preparing dataframe")
        data.drop_column('Timestamp')
        data.drop_column('Zip-code')
        if args.pre_process:
            print("Entries are all numbers...")
            data.all_numbers()

        # for feature importance calculation if needed
        x = data.x_values()
        y = data.y_values()

        data = data.values()


        train_data, test_data = train_test_split(data, test_size=args.ratio, random_state=42)

        print("Feature importance = NO")
        print("Converting tabular data into image...")
        if (args.feature_importance):
            print("Feature importance = YES")

            weights = feature_importance(x,y)

            if not args.eval_only:
                convert_to_images(train_data, weights, 'train')
                print("Data for training converted")
            convert_to_images(test_data, weights, 'test')
            print("Data for testing converted")


        else:
            if not args.eval_only:
                convert_to_images_raw(train_data, 'train')
                print("Data for training converted")
            convert_to_images_raw(test_data, 'test')
            print("Data for testing converted")

    # Initialize the model for this run with num_class = 5 since we predict the rating (1-5)
    model_ft, input_size = initialize_model(args.model_name, 5, args.feature_extract, use_pretrained=True)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join('/home/yucehan/Documents/BLG607/PredictwNN/images/', x), data_transforms[x]) for x in
                      ['train', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x
        in ['train', 'test']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Send the model to GPU
    model_ft = model_ft.to(device)

    if args.eval_only:
        name = input("Model name:")
        PATH = './models/' + name
        try:
            model_ft.load_state_dict(torch.load(PATH))
            res = test_model(model_ft, criterion, dataloaders_dict['test'])
            print('Test MSE Loss: {:.4f} '.format(res))
        except:
            raise Exception('Model does not exists!')
        exit()

    #  Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, args.lr, momentum=0.9)

    # Train
    model_ft, loss_hist, acc_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                 num_epochs=args.num_epochs)

    # testing
    #res = test_model(model_ft, criterion, dataloaders_dict['test'])
    #print('Test MSE Loss: {:.4f} '.format(res))

    # save the model
    name = args.model_name
    PATH = './models/' + name
    save = input("Save the model [y]/[N]: ")

    if save == 'y':
        torch.save(model_ft.state_dict(), PATH)
        print("Model Saved")

    # Post processing
    plt.title("MSE vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("MSE ")
    plt.plot(range(1, args.num_epochs + 1), loss_hist, label="Trained")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--lr", type=float, default=0.001, help="Discount Factor")
    parser.add_argument("--ratio", type=float, default=0.20, help="Train/test split")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--model_name", type=str, default='resnet', help="Name of the NN model")
    parser.add_argument("--feature_extract", type=bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--eval_only", type=bool, default=False)
    parser.add_argument("--clustering", type=bool, default=False)
    parser.add_argument("--feature_importance", type=bool, default=False)
    parser.add_argument("--pre_process", type=bool, default=False)
    parser.add_argument("--create_data", type=bool, default=False)

    args = parser.parse_args()

    main(args)
