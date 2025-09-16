import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd # Used for loading the CSV
import tiktoken
import random
import re
import unicodedata
from tqdm import tqdm
import os
import math
from enum import Enum

# ========================================================================================
# SECTION 1: Provided LTC Model Code (Unchanged)
# ========================================================================================
class MappingType(Enum):
    Identity = 0; Linear = 1; Affine = 2
class ODESolver(Enum):
    SemiImplicit = 0; Explicit = 1; RungeKutta = 2
class LTCCell(nn.Module):
    # ... (LTC Cell code remains the same as before) ...
    def __init__(self, input_size, num_units, solver=ODESolver.SemiImplicit, solver_clip=5.0):
        super(LTCCell, self).__init__(); self._input_size = input_size; self._num_units = num_units; self._solver = solver
        self._solver_clip = solver_clip; self._ode_solver_unfolds = 6; self._input_mapping = MappingType.Affine
        self._erev_init_factor = 1; self._w_init_max = 1.0; self._w_init_min = 0.01; self._cm_init_min = 0.5
        self._cm_init_max = 0.5; self._gleak_init_min = 1.0; self._gleak_init_max = 1.0; self._fix_cm = None
        self._fix_gleak = None; self._fix_vleak = None; self._w_min_value = 1e-5; self._w_max_value = 1000.0
        self._gleak_min_value = 1e-5; self._gleak_max_value = 1000.0; self._cm_t_min_value = 1e-6
        self._cm_t_max_value = 1000.0; self._get_variables(); self._map_inputs()
    @property
    def state_size(self): return self._num_units
    @property
    def output_size(self): return self._num_units
    def _map_inputs(self):
        if self._input_mapping in [MappingType.Affine, MappingType.Linear]:
            self.input_w = nn.Parameter(torch.Tensor(self._input_size)); init.constant_(self.input_w, 1.0)
        if self._input_mapping == MappingType.Affine:
            self.input_b = nn.Parameter(torch.Tensor(self._input_size)); init.constant_(self.input_b, 0.0)
    def _get_variables(self):
        self.sensory_mu = nn.Parameter(torch.Tensor(self._input_size, self._num_units)); self.sensory_sigma = nn.Parameter(torch.Tensor(self._input_size, self._num_units))
        self.sensory_W = nn.Parameter(torch.Tensor(self._input_size, self._num_units)); sensory_erev_init = (2 * np.random.randint(0, 2, size=[self._input_size, self._num_units]) - 1) * self._erev_init_factor
        self.sensory_erev = nn.Parameter(torch.from_numpy(sensory_erev_init).float()); init.uniform_(self.sensory_mu, a=0.3, b=0.8)
        init.uniform_(self.sensory_sigma, a=3.0, b=8.0); init.uniform_(self.sensory_W, a=self._w_init_min, b=self._w_init_max)
        self.mu = nn.Parameter(torch.Tensor(self._num_units, self._num_units)); self.sigma = nn.Parameter(torch.Tensor(self._num_units, self._num_units))
        self.W = nn.Parameter(torch.Tensor(self._num_units, self._num_units)); erev_init = (2 * np.random.randint(0, 2, size=[self._num_units, self._num_units]) - 1) * self._erev_init_factor
        self.erev = nn.Parameter(torch.from_numpy(erev_init).float()); init.uniform_(self.mu, a=0.3, b=0.8); init.uniform_(self.sigma, a=3.0, b=8.0)
        init.uniform_(self.W, a=self._w_init_min, b=self._w_init_max)
        if self._fix_vleak is None:
            self.vleak = nn.Parameter(torch.Tensor(self._num_units)); init.uniform_(self.vleak, a=-0.2, b=0.2)
        else: self.register_buffer('vleak', torch.full([self._num_units], self._fix_vleak))
        if self._fix_gleak is None:
            self.gleak = nn.Parameter(torch.Tensor(self._num_units))
            if self._gleak_init_max > self._gleak_init_min: init.uniform_(self.gleak, a=self._gleak_init_min, b=self._gleak_init_max)
            else: init.constant_(self.gleak, self._gleak_init_min)
        else: self.register_buffer('gleak', torch.full([self._num_units], self._fix_gleak))
        if self._fix_cm is None:
            self.cm_t = nn.Parameter(torch.Tensor(self._num_units))
            if self._cm_init_max > self._cm_init_min: init.uniform_(self.cm_t, a=self._cm_init_min, b=self._cm_init_max)
            else: init.constant_(self.cm_t, self._cm_init_min)
        else: self.register_buffer('cm_t', torch.full([self._num_units], self._fix_cm))
    def forward(self, inputs, state):
        if self._input_mapping in [MappingType.Affine, MappingType.Linear]: inputs = inputs * self.input_w
        if self._input_mapping == MappingType.Affine: inputs = inputs + self.input_b
        if self._solver == ODESolver.Explicit: next_state = self._ode_step_explicit(inputs, state)
        elif self._solver == ODESolver.SemiImplicit: next_state = self._ode_step_semi_implicit(inputs, state)
        elif self._solver == ODESolver.RungeKutta: next_state = self._ode_step_runge_kutta(inputs, state)
        else: raise ValueError(f"Unknown ODE solver '{str(self._solver)}'")
        return next_state, next_state
    def _ode_step_semi_implicit(self, inputs, state):
        v_pre = state; sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev; w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)
        for _ in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma); rev_activation = w_activation * self.erev
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator; v_pre = numerator / denominator
        return v_pre
    def _f_prime(self, inputs, state):
        v_pre = state; sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1); w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
        w_reduced_synapse = torch.sum(w_activation, dim=1); sensory_in = self.sensory_erev * sensory_w_activation; synapse_in = self.erev * w_activation
        sum_in = (torch.sum(sensory_in, dim=1) - v_pre * w_reduced_synapse + torch.sum(synapse_in, dim=1) - v_pre * w_reduced_sensory)
        f_prime = (1 / self.cm_t) * (self.gleak * (self.vleak - v_pre) + sum_in); return f_prime
    def _ode_step_explicit(self, inputs, state):
        v_pre = state; h = 0.1
        for _ in range(self._ode_solver_unfolds):
            f_prime = self._f_prime(inputs, v_pre); v_pre = v_pre + h * f_prime
            if self._solver_clip > 0: v_pre = torch.clamp(v_pre, -self._solver_clip, self._solver_clip)
        return v_pre
    def _ode_step_runge_kutta(self, inputs, state):
        v_pre = state; h = 0.1
        for _ in range(self._ode_solver_unfolds):
            k1 = h * self._f_prime(inputs, v_pre); k2 = h * self._f_prime(inputs, v_pre + 0.5 * k1)
            k3 = h * self._f_prime(inputs, v_pre + 0.5 * k2); k4 = h * self._f_prime(inputs, v_pre + k3)
            v_pre = v_pre + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if self._solver_clip > 0: v_pre = torch.clamp(v_pre, -self._solver_clip, self._solver_clip)
        return v_pre
    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = v_pre.unsqueeze(-1); mues = v_pre - mu; x = sigma * mues; return torch.sigmoid(x)
    def constrain_parameters(self):
        self.cm_t.data.clamp_(min=self._cm_t_min_value, max=self._cm_t_max_value)
        self.gleak.data.clamp_(min=self._gleak_min_value, max=self._gleak_max_value)
        self.W.data.clamp_(min=self._w_min_value, max=self._w_max_value)
        self.sensory_W.data.clamp_(min=self._w_min_value, max=self._w_max_value)

# ========================================================================================
# SECTION 2: Configuration
# ========================================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- IMPORTANT: Change this to your CSV file's name ---
DATA_PATH = "eng_french.csv"
MODEL_SAVE_PATH = "ltc_translator_best.pt"
NUM_EXAMPLES = 100000
BATCH_SIZE = 8
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
GRAD_CLIP = 1.0
VALIDATION_SPLIT = 0.1

print(f"Using device: {DEVICE}")

# ========================================================================================
# SECTION 3: Data Preprocessing
# ========================================================================================
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    # Ensure input is a string, handle potential float values from CSV
    w = str(w)
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    return w

# --- MODIFIED FOR CSV ---
def load_data(path, num_examples=None):
    """Loads data from a CSV file."""
    df = pd.read_csv(path)
    
    # Use specified number of examples if provided
    if num_examples:
        df = df.head(num_examples)
        
    # Define column names based on your file
    eng_col = "English words/sentences"
    fr_col = "French words/sentences"

    # Preprocess the sentences in each column
    df[eng_col] = df[eng_col].apply(preprocess_sentence)
    df[fr_col] = df[fr_col].apply(preprocess_sentence)

    # Convert the processed columns to a list of [english, french] pairs
    word_pairs = df[[eng_col, fr_col]].values.tolist()
    return word_pairs

# ========================================================================================
# SECTION 4: Tokenization (Unchanged)
# ========================================================================================
# class Tokenizer:
#     def __init__(self, encoding_name="cl100k_base"):
#         self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
#         self.tokenizer = tiktoken.get_encoding(
#             encoding_name,
#             # allowed_special=set(self.special_tokens)
#             allowed_special=set(self.special_tokens)
#         )
#         self.vocab_size = self.tokenizer.n_vocab
#         self.pad_id = self.tokenizer.encode("<pad>")[0]
#         self.sos_id = self.tokenizer.encode("<sos>")[0]
#         self.eos_id = self.tokenizer.encode("<eos>")[0]
#     def encode(self, s, add_special_tokens=True):
#         tokens = self.tokenizer.encode(s, allowed_special=set(self.special_tokens))
#         if add_special_tokens:
#             return [self.sos_id] + tokens + [self.eos_id]
#         return tokens
#     def decode(self, ids):
#         # Filter out special token IDs before decoding
#         special_token_ids = {self.pad_id, self.sos_id, self.eos_id}
#         ids_to_decode = [i for i in ids if i not in special_token_ids]
#         return self.tokenizer.decode(ids_to_decode)

class Tokenizer:
    """
    Lightweight wrapper around tiktoken that:
      - Uses a tiktoken encoding for ordinary text tokens.
      - Reserves a small block of IDs for custom special tokens:
        <pad> = 0, <sos> = 1, <eos> = 2, <unk> = 3
      - On encode: maps tiktoken ids -> (tiktoken_id + OFFSET).
      - On decode: maps ids >= OFFSET back to tiktoken ids before decoding.
    """
    def __init__(self, encoding_name="cl100k_base"):
        import tiktoken  # keep local import clarity
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self._encoding = tiktoken.get_encoding(encoding_name)  # no allowed_special here
        self._base_vocab_size = self._encoding.n_vocab
        # reserve initial ids for specials
        self._offset = len(self.special_tokens)
        # special token ids
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        # full vocab size = tiktoken vocab + reserved specials
        self.vocab_size = self._base_vocab_size + self._offset

    def encode(self, s, add_special_tokens=True):
        # ensure string input
        s = "" if s is None else str(s)
        # encode with tiktoken (ordinary tokens)
        base_tokens = self._encoding.encode(s)
        # offset the token ids so they don't collide with special ids
        token_ids = [t + self._offset for t in base_tokens]
        if add_special_tokens:
            return [self.sos_id] + token_ids + [self.eos_id]
        return token_ids

    def decode(self, ids):
        # ids may contain special tokens; remove pad/sos/eos before decoding
        # take only ids that correspond to base tokens (>= offset)
        base_ids = [i - self._offset for i in ids if i >= self._offset]
        if not base_ids:
            return ""
        # decode using tiktoken
        return self._encoding.decode(base_ids)


# ========================================================================================
# SECTION 5: Dataset and DataLoader (Unchanged)
# ========================================================================================
class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer_en, tokenizer_fr):
        self.pairs = pairs
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        eng_sentence, fra_sentence = self.pairs[idx]
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

# ========================================================================================
# SECTION 6: Seq2Seq Model with LTC (Unchanged)
# ========================================================================================
class EncoderLTC(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super(EncoderLTC, self).__init__(); self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.ltc_cell = LTCCell(embed_size, hidden_size)
    def forward(self, x):
        seq_length, batch_size = x.shape
        hidden_state = torch.zeros(batch_size, self.hidden_size).to(x.device)
        embedded = self.embedding(x)
        for t in range(seq_length):
            hidden_state, _ = self.ltc_cell(embedded[t], hidden_state)
        return hidden_state

class DecoderLTC(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super(DecoderLTC, self).__init__(); self.hidden_size = hidden_size
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
    def __init__(self, encoder, decoder, target_vocab_size, device):
        super(Seq2SeqLTC, self).__init__(); self.encoder = encoder; self.decoder = decoder
        self.target_vocab_size = target_vocab_size; self.device = device
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

# ========================================================================================
# SECTION 7: Training & Evaluation (Unchanged)
# ========================================================================================
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def train_epoch(model, dataloader, optimizer, criterion, clip, teacher_force_ratio):
    model.train(); epoch_loss = 0
    for _, (src, trg) in enumerate(tqdm(dataloader, desc="Training")):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg, teacher_force_ratio=teacher_force_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim); trg = trg[1:].view(-1)
        loss = criterion(output, trg); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip); optimizer.step()
        model.constrain_parameters()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval(); epoch_loss = 0
    with torch.no_grad():
        for _, (src, trg) in enumerate(tqdm(dataloader, desc="Evaluating")):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, teacher_force_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim); trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# ========================================================================================
# SECTION 8: Main Execution (Unchanged logic, just check DATA_PATH)
# ========================================================================================
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file '{DATA_PATH}' not found."); exit()
    
    print("Loading and preprocessing data from CSV...")
    pairs = load_data(DATA_PATH, num_examples=NUM_EXAMPLES)
    
    print("Initializing tokenizers...")
    tokenizer_en = Tokenizer(); tokenizer_fr = Tokenizer()
    
    dataset = TranslationDataset(pairs, tokenizer_en, tokenizer_fr)
    
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    collate_with_pad = lambda batch: collate_fn(batch, tokenizer_fr.pad_id)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_with_pad)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_with_pad)
    
    print("Initializing model...")
    INPUT_DIM = tokenizer_en.vocab_size; OUTPUT_DIM = tokenizer_fr.vocab_size
    encoder = EncoderLTC(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    decoder = DecoderLTC(OUTPUT_DIM, EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    model = Seq2SeqLTC(encoder, decoder, OUTPUT_DIM, DEVICE).to(DEVICE)
    
    model.apply(initialize_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_fr.pad_id)
    
    print("Starting training...")
    best_valid_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        teacher_force_ratio = 1.0 - (epoch / NUM_EPOCHS)
        
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, GRAD_CLIP, teacher_force_ratio)
        valid_loss = evaluate(model, val_dataloader, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"----> New best model saved to {MODEL_SAVE_PATH}")

        print(f"Epoch: {epoch+1:02} | Teacher Forcing: {teacher_force_ratio:.2f}")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")