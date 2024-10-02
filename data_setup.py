import datasets
import pandas 
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch


def data_setup(models:str,
                tokenizer:str,
                types:str ):
    
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
    data = rawdf_class[types].to_pandas()[['SMILES','logBB']]

    # FEATURINZING X columns
    smiles= data['SMILES'].tolist()
    inputs = tokenizer(smiles, padding=True, truncation=True, return_tensors='pt', max_length=60)
    with torch.no_grad():
        embeddings = model(**inputs)[0][::,:,::]
        embeddings_squeez = embeddings.unsqueeze(dim=0)
        embed_x =embeddings_squeez.permute(1,0,2,3)

    # Response column y

    y =pandas.cut(x=data['logBB'], bins=[-2.80 , 0.016 , 1.8], labels=['BBB-', 'BBB+'])
    y = [0 if i=='BBB-' else 1 for i in y ]

    return  embed_x,y
    
 