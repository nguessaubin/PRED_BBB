# IMPORT DATA
import datasets
import pandas 
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
from sklearn.model_selection import train_test_split
from torch import nn

#any model weights from the link above will work here
model = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
tokenizer_chembert = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Load Dataset from hugging face server
rawdf_class = datasets.load_dataset("maomlab/B3DB",name="B3DB_regression")

# Extract X_train
X_train = rawdf_class['train'].to_pandas()[['SMILES','logBB']]

# Get y for classification
X_train['logBB_class'] = pandas.cut(x=X_train['logBB'], bins=[-2.80 , 0.016 , 1.8], labels=['BBB-', 'BBB+'])


def calculate_embeddings(df, model, tokenizer):
    smiles= df.iloc[60:100]['SMILES'].tolist()
    inputs = tokenizer(smiles, padding=True, truncation=True, return_tensors='pt', max_length=120)
    with torch.no_grad():
        embeddings = model(**inputs)[0][::,:,::]
    embeddings_squeez = embeddings.unsqueeze(dim=0)
    embed_x =embeddings_squeez.permute(1,0,2,3)
    return embed_x 

# GET X AND Y
X= calculate_embeddings(X_train, model, tokenizer_chembert)
y = X_train['logBB_class']
y= [0 if i=='BBB-' else 1 for i in y ]

# MODEL IMPLEMENTATION

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    shuffle=True) # make the random split reproducible


# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

## 1. Construct a model class that subclasses nn.Module
class CONVV(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack1 = nn.Sequential(
            nn.Conv2d(1, 3,3,2), # takes in 2 features (X), produces 5 features
            nn.ReLU(),
            nn.Conv2d(3, 1,3,2), # takes in 5 features, produces 1 feature (y)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=573, out_features=1))
        
        self.stack2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), # takes in 2 features (X), produces 5 features
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=1))
            
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return  self.stack1(x) #computation goes through layer_1 first then the output of layer_1 goes through layer_2

# 4. Create an instance of the model and send it to target device
model_0 = CONVV().to(device)


# Fit the model
torch.manual_seed(42)
epochs = 1000

# Put all data on target device
#y_train = X_train.to(device), y_train.to(device)
#X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    
        
    # 1. Forward pass
    y_logits = model_0(X_train)
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_pred, torch.Tensor(y_train).unsqueeze(dim=1))# BCEWithLogitsLoss calculates loss using logits
     
    # 3. Optimizer zero grad
    optimizer.zero_grad()
        
    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test)
        test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
      # 2. Calcuate loss and accuracy
        test_loss = loss_fn( test_pred,torch.Tensor(y_test).unsqueeze(dim=1))

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}| Test Loss: {test_loss:.5f}")