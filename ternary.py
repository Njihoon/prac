import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from torchvision import datasets, transforms
#from torchsummary import summary


def ParseArgs():
    parser = argparse.ArgumentParser(
        description='Example Ternary Weights Network (Pytorch)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='batch size for training(default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='batch size for testing(default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epoch to train(default: 100)')
    parser.add_argument('--lr-epochs', type=int, default=20, metavar='N',
                        help='number of epochs to decay learning rate(default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate(default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum(default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-5, metavar='WD',
                        help='weight decay(default: 1e-5)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed(default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def train(args, epoch_index, train_loader, model, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_index, batch_idx * len(data), len(train_loader.dataset),
                             100. * batch_idx / len(train_loader), loss.data[0]))


def test(args, model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(test_loader.dataset)

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return acc


def adjust_learning_rate(learning_rate, optimizer, epoch_index, lr_epoch):
    lr = learning_rate * (0.1 ** (epoch_index // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr


def ternarize(tensor):
    """
    Ternarize by alpha / delta?
    1. ternarize by > or <
    2. ternarize by sign
    :param tensor:
    :return:
    """
    output = torch.zeros(tensor.size()).type(torch.FloatTensor)
    tensor_delta = delta(tensor)
    tensor_alpha = alpha(tensor, tensor_delta)
    for i in range(tensor.size()[0]):
        for w in tensor[i].view(1, -1):
            pos_one = (w > tensor_delta[i]).type(torch.FloatTensor)
            neg_one = -1 * (w < -tensor_delta[i]).type(torch.FloatTensor)
        out = torch.add(pos_one, neg_one).view(tensor.size()[1:])
        output[i] = torch.add(output[i], torch.mul(out, tensor_alpha[i]))  # multiply for float?

    return output


def binarize(tensor):
    output = torch.zeros(tensor.size()).type(torch.IntTensor)
    for i in range(tensor.size()[0]):
        for w in tensor[i].view(1, -1):
            pos_one = (w > 0).type(torch.FloatTensor)
            neg_one = -1 * (w < 0).type(torch.FloatTensor)
        out = torch.add(pos_one, neg_one).view(tensor.size()[1:])
        output[i] = torch.add(output[i], out.type(torch.IntTensor))
    return output


def alpha(tensor, delta):
    alpha_list = []
    for i in range(tensor.size()[0]):
        truth_value = [0]
        absvalue = tensor[i].view(1, -1).abs()
        for w in absvalue:
            truth_value = w > delta[i]  # print to see
        count = truth_value.sum()
        abssum = torch.matmul(absvalue, truth_value.type(absvalue.dtype).view(-1, 1))
        alpha_list.append(abssum / count)
    alpha = alpha_list[0]
    for i in range(len(alpha_list) - 1):
        alpha = torch.cat((alpha, alpha_list[i + 1]))
    return alpha


def delta(tensor, delta=0.7):
    n = tensor[0].nelement()
    if len(tensor.size()) == 4:  # convolution layer
        return delta * tensor.norm(1, 3).sum(2).sum(1).div(n)
    elif len(tensor.size()) == 2:  # fc layer
        return delta * tensor.norm(1, 1).div(n)
    else:
        raise Exception("Delta ERROR!")


class TernaryLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(TernaryLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        # self.weight.data = ternarize(self.weight.data).type(torch.float32)
        out = F.linear(input, self.weight, self.bias)
        return ternarize(out)


class TernaryConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(TernaryConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        # self.weight.data = ternarize(self.weight.data).type(torch.float32)
        out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return ternarize(out)


class LeNet5_T(nn.Module):
    def __init__(self):
        super(LeNet5_T, self).__init__()
        self.conv1 = TernaryConv2d(1, 32, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = TernaryConv2d(32, 64, kernel_size=5)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = TernaryLinear(1024, 512)
        self.fc2 = TernaryLinear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(self.bn_conv1(x), 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(self.bn_conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5,))
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5,))
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(self.bn_conv1(x), 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(self.bn_conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AlexNet_T(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """

    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            TernaryConv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            TernaryConv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            TernaryConv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            TernaryConv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            TernaryConv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            TernaryLinear(in_features=(256 * 6 * 6), out_features=4096),
            nn.Dropout(p=0.5, inplace=True),
            TernaryLinear(in_features=4096, out_features=4096),
            TernaryLinear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)


class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)


if __name__ == '__main__':
    args = ParseArgs()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    # Train dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data', train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

    model = LeNet5()
    model_T = LeNet5_T()

    summary(model, input_size=(1, 28, 28))
    summary(model_T, input_size=(1, 28, 28))

    if args.cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()
    # optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_acc = 0.0
    for epoch_index in range(1, args.epochs + 1):
        adjust_learning_rate(learning_rate, optimizer, epoch_index, args.lr_epochs)
        train(args, epoch_index, train_loader, model, optimizer, criterion)
        acc = test(args, model, test_loader, criterion)
        if acc > best_acc:
            best_acc = acc
            # U.save_model(model, best_acc)

# test = np.random.normal(size=(10, 20))
test = np.random.randint(size=(3, 3, 227, 227), low=0, high=255) / 255
tensor_test = torch.from_numpy(test.astype(np.float32))

model = AlexNet_T()
model.forward(tensor_test)

x = torch.from_numpy(test)
ternarize(x)
