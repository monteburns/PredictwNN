from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier


def feature_importance(X, y):
    # define the model
    model = DecisionTreeClassifier()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_

    return importance


def convert_to_images(array, weights, folder):
    i = 0
    PATH = '/home/yucehan/Documents/BLG607/PredictwNN'
    font = []

    for i in range(len(weights)):
        font.append(ImageFont.truetype("arial.ttf", size=int(160 * weights[i])))

    for row in array:
        background = np.array([[0 for _ in range(224)] for _ in range(224)], dtype='uint8')
        image = Image.fromarray(background)
        draw = ImageDraw.Draw(image)
        draw.text((10, 50), str("%03d" % row[0]), fill='white', font=font[0])
        draw.text((110, 50), str("%03d" % row[1]), fill='white', font=font[1])
        draw.text((180, 50), str("%03d" % row[2]), fill='white', font=font[2])
        draw.text((10, 150), str("%03d" % row[3]), fill='white', font=font[3])
        draw.text((110, 150), str("%03d" % row[4]), fill='white', font=font[4])

        image.save(PATH + '/images/' + folder + '/' + str(row[5]) + '/' + str(i) + '.png')
        i = i + 1

    return True

def convert_to_images_raw(array, folder):
    i=0
    PATH = '/home/yucehan/Documents/BLG607/PredictwNN'

    font = ImageFont.truetype("arial.ttf", size=30)

    for row in array:

        image = Image.new('RGBA', (224, 224), (0, 0, 0))

        draw = ImageDraw.Draw(image)
        draw.text((10, 50), str(row[0]), fill='white', font=font)
        draw.text((110, 50), str(row[1]), fill='white', font=font)
        draw.text((180, 50), str(row[2]), fill='white', font=font)
        draw.text((10, 150), str(row[3]), fill='white', font=font)
        draw.text((100, 100), str(row[4]), fill='white', font=font)

        image.save(PATH+'/images/'+ folder + '/' + str(row[5]) + '/' +str(i)+'.png')
        i=i+1

    return True


def test_model(model, criterion, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)


        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    return epoch_loss
