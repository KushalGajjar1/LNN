import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
import argparse


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = str(w)
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r'[" "]+', " ", w)
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
        
        self.ltc_cell_fw = LTCCell(embed_size, hidden_size)
        self.ltc_cell_bw = LTCCell(embed_size, hidden_size)
        
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        seq_length, batch_size = x.shape
        
        hidden_state_fw = torch.zeros(batch_size, self.hidden_size).to(x.device)
        outputs_fw = []
        embedding = self.embedding(x)
        for t in range(seq_length):
            hidden_state_fw, _ = self.ltc_cell_fw(embedding[t], hidden_state_fw)
            outputs_fw.append(hidden_state_fw)
        
        hidden_state_bw = torch.zeros(batch_size, self.hidden_size).to(x.device)
        outputs_bw = []
        for t in range(seq_length - 1, -1, -1):
            hidden_state_bw, _ = self.ltc_cell_bw(embedding[t], hidden_state_bw)
            outputs_bw.append(hidden_state_bw)
        outputs_bw = outputs_bw[::-1]
        
        outputs_fw = torch.stack(outputs_fw)
        outputs_bw = torch.stack(outputs_bw)
        
        encoder_outputs = torch.cat((outputs_fw, outputs_bw), dim=2)
        final_hidden = torch.cat((hidden_state_fw, hidden_state_bw), dim=1)
        decoder_initial_hidden = torch.tanh(self.fc(final_hidden))

        return encoder_outputs, decoder_initial_hidden

    def constrain_parameters(self):
        self.ltc_cell_fw.constrain_parameters()
        self.ltc_cell_bw.constrain_parameters()


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))
        
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)


class DecoderLTC(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super(DecoderLTC, self).__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(hidden_size, hidden_size)
        self.embedding = nn.Embedding(output_size, embed_size)
        
        self.ltc_cell = LTCCell(embed_size + (hidden_size * 2), hidden_size)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, encoder_outputs):
        attention_weights = self.attention(hidden_state, encoder_outputs).unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        context_vector = torch.bmm(attention_weights, encoder_outputs).permute(1, 0, 2)
        
        x = x.unsqueeze(0)
        embedded = self.embedding(x)
        
        ltc_input = torch.cat((embedded, context_vector), dim=2)
        
        output, hidden_state = self.ltc_cell(ltc_input.squeeze(0), hidden_state)
        
        predictions = self.fc(output)
        
        return predictions, hidden_state

    def constrain_parameters(self):
        self.ltc_cell.constrain_parameters()



class Seq2SeqLTC(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqLTC, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_force_ratio=0.5):
        trg_len, batch_size = target.shape
        target_vocab_size = self.decoder.fc.out_features
        outputs_list = []
        
        encoder_outputs, hidden_state = self.encoder(source)
        
        x = target[0, :]
        for t in range(1, trg_len):
            output, hidden_state = self.decoder(x, hidden_state, encoder_outputs)
            outputs_list.append(output)
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        
        if len(outputs_list) == 0:
            outputs = torch.zeros(trg_len, batch_size, target_vocab_size, device=self.device)
        else:
            first = torch.zeros(batch_size, target_vocab_size, device=self.device)
            outputs = torch.stack([first] + outputs_list, dim=0)
            if outputs.shape[0] < trg_len:
                pad_shape = (trg_len - outputs.shape[0], batch_size, target_vocab_size)
                pad_tensor = torch.zeros(pad_shape, device=self.device)
                outputs = torch.cat([outputs, pad_tensor], dim=0)

        return outputs

    def constrain_parameters(self):
        self.encoder.constrain_parameters()
        self.decoder.constrain_parameters()


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
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        model.constrain_parameters()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
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
    parser.add_argument('--model_save_path', default="ltc_translator_attention.pt", type=str)
    parser.add_argument('--num_examples', default=100000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--embed_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--validation_split', default=0.1, type=float)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.data_path):
        print(f"Error: Data file '{args.data_path}' not found.")
        exit()

    print("Loading and preprocessing data...")
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

    print("Initializing model with Attention...")
    INPUT_DIM = tokenizer_en.vocab_size
    OUTPUT_DIM = tokenizer_fr.vocab_size
    
    encoder = EncoderLTC(INPUT_DIM, args.embed_size, args.hidden_size).to(device)
    decoder = DecoderLTC(OUTPUT_DIM, args.embed_size, args.hidden_size).to(device)
    model = Seq2SeqLTC(encoder, decoder, device).to(device)

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