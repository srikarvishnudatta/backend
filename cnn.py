import torch.nn as nn
from torch.nn.functional import relu, softmax

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        '''
        Layer 1
        Converting 1 channel grayscale input and feed it to layer 1.
        Layers 1 converts it to feature map of 64 (64 different features).
        Using padding as 2 (essentially adds two rows and columns of 0 to input image matrix)
        So if our image size is 48*48, it will be 50*50. (each pixel has value between 0 to 255)
        0 represents darker and 255 represents lighter.
        '''
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        '''
        Using max pooling layer of 2*2 with stride of 2 (every sub matrix will be slide by 2)
        '''
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        '''
            Droping few subset of neurons from the computation.
            This ensure not to rely too much on any specific set of neurons.
            In our case we take probability of 0.3 so 0.3% of set of neurons will be dropped.
        '''
        self.dropout1 = nn.Dropout(0.3)
        # self.pool11 = nn.MaxPool2d(2, 2)

        # Layer 3
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        '''
        Output of layer 3 will be fed to normalization layer with 16 as number of input channel.
        Normalization layer will normalize each of these 16 channels independently.
        '''
        self.bn1 = nn.BatchNorm2d(16)

        # Layer 4
        self.conv4 = nn.Conv2d(16, 4, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(4)

        # self.conv5 = nn.Conv2d(8, 4, kernel_size=5, padding=2)
        # self.pool4 = nn.MaxPool2d(2,2)
        # self.bn3 = nn.BatchNorm2d(4)

        '''
        Full connected layer (Linear layer)
        4 is the number of output channels
        6 * 6 is the spatial dimensions of the feature map after the last pooling layer.
        It also applies weight and biases which will be learned during training.
        4 represents total number of classifications that we are trying to achieve.
        In our case it's Angry, Focused, Neutral, Tired.
        '''
        self.fc = nn.Linear(4 * 6 * 6, 4)

    '''
        Input x (batch_size, input_channel, height, width) (32, 1, 48, 48)
    '''
    def forward(self, x):
        # Output x: (32, 64, new_height, new_width) (new height and width depends on kernel size, padding, and stride)
        x = relu(self.conv1(x))
        # Output x: (32, 64, new_height, new_width)
        x = self.pool1(x)

        # Output x: (32, 32, new_height, new_width)
        x = relu(self.conv2(x))
        x = self.dropout1(x)
        # x = self.pool11(x)

        # Output x: (32, 16, new_height, new_width)
        x = relu(self.conv3(x))
        x = self.pool2(x)
        x = self.bn1(x)

        # Output x: (32, 4, new_height, new_width)
        x = relu(self.conv4(x))
        x = self.pool3(x)

        # Output x: (32, 4, new_height, new_width)
        x = self.bn2(x)

        # x = relu(self.conv5(x))
        # x = self.pool4(x)
        # x = self.bn3(x)

        # print(x.size())

        # Output x: (32, 4 * 6 * 6) 4 channels with (6*6) feature map.
        x = x.view(-1, 4 * 6 * 6)

        # Input (32, 4 * 6 * 6)
        # Output: (32, 4) => 4 different classes in our case.
        x = self.fc(x)

        # Softmax is the activation function, which converts the raw output scores into probabilities for each class for classification.
        return softmax(x, dim=1)