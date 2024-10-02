
import data_loader
import torch
from torch import nn
import model_builder
import engine

import model_builder
device = "cuda" if torch.cuda.is_available() else "cpu"


# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 0.001
WORKERS = 0

# Instantiate an instance of the model from the "model_builder.py" script
torch.manual_seed(42)
model = model_builder.SimpleCNN(input_shape=1,
                                output_shape=3).to(device)

# Setup loss and optimizer 
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

engine.train(model=model,
             train_dataloader=data_loader.Custum_dataloader('train',  batch_size=BATCH_SIZE, num_workers=WORKERS),
             test_dataloader=data_loader.Custum_dataloader('test', batch_size=BATCH_SIZE, num_workers=WORKERS),
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)