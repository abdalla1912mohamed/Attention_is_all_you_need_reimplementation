import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset,DataLoader,random_split
from pathlib import Path
from dataset import BilingualDataset,casual_mask
import torch.utils
from model import build_trasformer
from config import get_weights_file_path,get_config
from tqdm import tqdm
from train import get_dataset,run_validation,get_model


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

config=get_config()
train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_dataset(config)
model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)

model_filename=get_weights_file_path(config,config['preload'])
print(f'Preloading model {model_filename}')
state=torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])


run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,config['seq_len'],device,lambda x: print(x),0,None,numexamples=10)
