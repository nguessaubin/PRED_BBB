import torch
from torch import nn

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self,input_shape:int,output_shape:int)->None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, output_shape,3,2), # takes in 2 features (X), produces 5 features
            nn.ReLU(),
            nn.Conv2d(3, 1,3,2), # takes in 5 features, produces 1 feature (y)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=573, out_features=1))
      
       # self.conv_block_2 = nn.Sequential(
       #    nn.Conv2d(3, hidden_units, kernel_size=3, padding=0),
       #    nn.ReLU(),
       #    nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
       #    nn.ReLU(),
       #    nn.MaxPool2d(2))

       # self.classifier = nn.Sequential(
       #    nn.Flatten(),
       #    # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
        #    nn.Linear(in_features=hidden_units*13*13,
        #            out_features=output_shape))
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # x = self.conv_block_2(x)
        # x = self.classifier(x)
        return x



