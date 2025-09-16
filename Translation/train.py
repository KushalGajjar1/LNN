import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from ltc_model import LTCCell

import numpy as np
import pandas as pd
import tiktoken
import random
import re
import unicodedata
from tqdm import tqdm
import os
import math
from enum import Enum
import argparse


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = str(w)
    w = unicode_to_ascii(w.lower().strip())
    # w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    return w


def load_data(path, num_examples=None):

    df = pd.read_csv(path)
    if num_examples:
        df = df.head(num_examples)
    
    eng_col = "English words/sentences"
    fr_col = "French words/sentences"

    df[eng_col] = df[eng_col].apply(preprocess_sentence)
    df[fr_col] = df[fr_col].apply(preprocess_sentence)

    word_pairs = df[[eng_col, fr_col]].values.tolist()
    return word_pairs


class Tokenizer:

    def __init__(self, encoding_name="cl100k_base"):
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._base_vocab_size = self._encoding.n_vocab
        self._offset = len(self.special_tokens)
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.vocab_size = self._base_vocab_size + self._offset

    def encode(self, s, add_special_tokens=True):
        s = "" if s is None else str(s)
        base_tokens = self._encoding.encode(s)
        token_ids = [t + self._offset for t in base_tokens]
        if add_special_tokens:
            return [self.sos_id] + token_ids + [self.eos_id]
        return token_ids
    
    def decode(self, ids):
        base_ids = [i - self._offset for i in ids if i >= self._offset]
        if not base_ids:
            return ""
        return self._encoding.decode(base_ids)
    

class TranslationDataset(Dataset):

    def __init__(self, pairs, tokenizer_en, tokenizer_fr):
        self.pairs = pairs
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        eng_sentence, fra_sentence = self.pairs[index]
        eng_tokens = self.tokenizer_en.encode(eng_sentence)
        fra_tokens = self.tokenizer_fr.encode(fra_sentence)
        return torch.tensor(eng_tokens, dtype=torch.long), torch.tensor(fra_tokens, dtype=torch.long)
    


def collate_fn(batch, pad_id):
    eng_batch, fra_batch = [], []
    for eng_item, fra_item in batch:
        eng_batch.append(eng_item)
        fra_batch.append(fra_item)
    eng_padded = pad_sequence(eng_batch, batch_first=False, padding_value=pad_id)
    fra_padded = pad_sequence(fra_batch, batch_first=False, padding_value=pad_id)
    return eng_padded, fra_padded


class EncoderLTC(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size):
        super(EncoderLTC, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.ltc_cell = LTCCell(embed_size, hidden_size)

    def forward(self, x):
        seq_length, batch_size = x.shape
        hidden_state = torch.zeros(batch_size, self.hidden_size).to(x.device)
        embedding = self.embedding(x)
        for t in range(seq_length):
            hidden_state, _ = self.ltc_cell(embedding[t], hidden_state)
        return hidden_state
    

class DecoderLTC(nn.Module):

    def __init__(self, output_size, embed_size, hidden_size):
        super(DecoderLTC, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embed_size)
        self.ltc_cell = LTCCell(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        x = x.unsqueeze(0)
        embedded = self.embedding(x)
        output, hidden_state = self.ltc_cell(embedded[0], hidden_state)
        predictions = self.fc(output)
        return predictions, hidden_state
    

class Seq2SeqLTC(nn.Module):

    def __init__(self, encoder, decoder, taregt_vocab_size, device):
        super(Seq2SeqLTC, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = taregt_vocab_size
        self.device = device

    def forward(self, source, target, teacher_force_ratio=0.5):
        trg_len, batch_size = target.shape
        outputs = torch.zeros(trg_len, batch_size, self.target_vocab_size).to(self.device)
        hidden_state = self.encoder(source)
        x = target[0, :]
        for t in range(1, trg_len):
            output, hidden_state = self.decoder(x, hidden_state)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs
    def constrain_parameters(self):
        self.encoder.ltc_cell.constrain_parameters()
        self.decoder.ltc_cell.constrain_parameters()
    

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train_epoch(model, dataloader, optimizer, criterion, clip, teacher_force_ratio):
    model.train()
    epoch_loss = 0.0
    for _, (src, trg) in enumerate(tqdm(dataloader, desc="Training")):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_force_ratio=teacher_force_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        model.constrain_parameters()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval(); epoch_loss = 0
    with torch.no_grad():
        for _, (src, trg) in enumerate(tqdm(dataloader, desc="Evaluating")):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_force_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default="eng_french.csv", type=str)
    parser.add_argument('--model_save_path', default="ltc_translator_best.pt", type=str)
    parser.add_argument('--num_examples', default=20000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--embed_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--validation_split', default=0.1, type=float)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.data_path):
        print(f"Error: Data file '{args.data_path}' not found.")
        exit()

    print("Loading and preprocessing data from CSV...")
    pairs = load_data(args.data_path, num_examples=args.num_examples)

    print("Initializing tokenizers...")
    tokenizer_en = Tokenizer()
    tokenizer_fr = Tokenizer()

    dataset = TranslationDataset(pairs, tokenizer_en, tokenizer_fr)

    val_size = int(len(dataset) * args.validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    collate_with_pad = lambda batch: collate_fn(batch, tokenizer_fr.pad_id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_pad)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_pad)

    print("Initializing model...")
    INPUT_DIM = tokenizer_en.vocab_size
    OUTPUT_DIM = tokenizer_fr.vocab_size
    encoder = EncoderLTC(INPUT_DIM, args.embed_size, args.hidden_size).to(device)
    decoder = DecoderLTC(OUTPUT_DIM, args.embed_size, args.hidden_size).to(device)
    model = Seq2SeqLTC(encoder, decoder, OUTPUT_DIM, device).to(device)

    model.apply(initialize_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_fr.pad_id)

    print("Starting training...")
    best_valid_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        teacher_force_ratio = 1.0 - (epoch / args.num_epochs)
        
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, args.grad_clip, teacher_force_ratio)
        valid_loss = evaluate(model, val_dataloader, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"----> New best model saved to {args.model_save_path}")

        print(f"Epoch: {epoch+1:02} | Teacher Forcing: {teacher_force_ratio:.2f}")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")