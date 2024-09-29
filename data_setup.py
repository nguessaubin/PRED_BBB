import datasets
import pandas 
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import pickle5 as pickle

def data_setup(models:str,
               tokenizer:str,
               type:str ):
    
    ''' 
    Create X and y in regard of data type ("Train, and Test) for featurizing.
    models: 'seyonec/ChemBERTa-zinc-base-v1'
    tokenizer : 'seyonec/ChemBERTa-zinc-base-v1'
    type: 'Train or Test'
    '''
    #any model weights from the link above will work here
    model = AutoModelWithLMHead.from_pretrained(models)
    tokenizer= AutoTokenizer.from_pretrained(tokenizer)

    # Load Dataset from hugging face server
    rawdf_class = datasets.load_dataset("maomlab/B3DB",name="B3DB_regression")

    # Extract data
    data = rawdf_class[type].to_pandas()[['SMILES','logBB']]

    # FEATURINZING
    smiles= data['SMILES'].tolist()
    inputs = tokenizer(smiles, padding=True, truncation=True, return_tensors='pt', max_length=60)
    with torch.no_grad():
        embeddings = model(**inputs)[0][::,:,::]
        embeddings_squeez = embeddings.unsqueeze(dim=0)
        embed_x =embeddings_squeez.permute(1,0,2,3)
    return embed_x

#    torch.save(data_setup(models, tokenizer), 'Name.pt')


