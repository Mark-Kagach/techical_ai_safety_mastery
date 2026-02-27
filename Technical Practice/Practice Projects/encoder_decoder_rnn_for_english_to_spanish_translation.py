# -*- coding: utf-8 -*-

# Data

# imports, device
!pip install torchmetrics
import torch
import tokenizers
import transformers
import torchmetrics
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device

# dataset load and split
nmt_original_valid_set, nmt_test_set = load_dataset(path="ageron/tatoeba_mt_train", name="eng-spa", split=["validation", "test"])
split = nmt_original_valid_set.train_test_split(train_size=0.8, seed=69)
nmt_train_set, nmt_valid_set = split["train"], split["test"]

nmt_train_set[0]

# bpe tokenizer for both english and spanish.
def train_eng_spa():
  for pair in nmt_train_set:
    yield pair["source_text"]
    yield pair["target_text"]

max_length = 256
vocab_size = 10_000
nmt_tokenizer_model = tokenizers.models.BPE(unk_token="<unk>")
nmt_tokenizer = tokenizers.Tokenizer(nmt_tokenizer_model)
nmt_tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
nmt_tokenizer.enable_truncation(max_length=max_length)
nmt_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
nmt_tokenizer_trainer = tokenizers.trainers.BpeTrainer( vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
nmt_tokenizer.train_from_iterator(train_eng_spa(), nmt_tokenizer_trainer)

nmt_tokenizer.encode("i dare you, i double dare you motherfucker").ids

nmt_tokenizer.encode("te reto, te reto doblemente hijo de puta").ids

# put together tokenized english and its corresponding targets
from collections import namedtuple
fields = ["src_token_ids", "src_mask", "tgt_token_ids", "tgt_mask"]
class NmtPair(namedtuple("NmtPairBase", fields)):
  def to(self, device):
    return NmtPair(self.src_token_ids.to(device), self.src_mask.to(device), self.tgt_token_ids.to(device), self.tgt_mask.to(device))

# data loaders
def nmt_collate_fn(batch):
  src_texts = [pair['source_text'] for pair in batch]
  tgt_texts = [f"<s> {pair['target_text']} </s>" for pair in batch]
  src_encodings = nmt_tokenizer.encode_batch(src_texts)
  tgt_encodings = nmt_tokenizer.encode_batch(tgt_texts)
  src_token_ids = torch.tensor([enc.ids for enc in src_encodings])
  tgt_token_ids = torch.tensor([enc.ids for enc in tgt_encodings])
  src_mask = torch.tensor([enc.attention_mask for enc in src_encodings])
  tgt_mask = torch.tensor([enc.attention_mask for enc in tgt_encodings])
  inputs = NmtPair(src_token_ids, src_mask, tgt_token_ids[:, :-1], tgt_mask[:, :-1])
  labels = tgt_token_ids[:, 1:]
  return inputs, labels

batch_size = 32
nmt_train_loader = DataLoader(nmt_train_set, batch_size=batch_size, collate_fn=nmt_collate_fn, shuffle=True)
nmt_valid_loader = DataLoader(nmt_valid_set, batch_size=batch_size, collate_fn=nmt_collate_fn)
nmt_test_loader = DataLoader(nmt_test_set, batch_size=batch_size, collate_fn=nmt_collate_fn)

"""# Model"""

# define training and metrics function
# (reusing train and metrics functions from past notebooks: )
def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()

def train(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs, patience=2, factor=0.5, epoch_callback=None):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=patience, factor=factor)
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.0
        metric.reset()
        model.train()
        if epoch_callback is not None:
            epoch_callback(model, epoch)
        for index, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
            train_metric = metric.compute().item()
            print(f"\rBatch {index + 1}/{len(train_loader)}", end="")
            print(f", loss={total_loss/(index+1):.4f}", end="")
            print(f", {train_metric=:.2%}", end="")
        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(train_metric)
        val_metric = evaluate_tm(model, valid_loader, metric).item()
        history["valid_metrics"].append(val_metric)
        scheduler.step(val_metric)
        print(f"\rEpoch {epoch + 1}/{n_epochs},                      "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.2%}, "
              f"valid metric: {history['valid_metrics'][-1]:.2%}")
    return history

# custom module with defined model and forward
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NmtModel(nn.Module):
  def __init__(self, vocab_size, embed_dim=512, pad_id=0, hidden_dim=512, n_layers=2):
    super().__init__()
    self.embed=nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
    self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
    self.decoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
    self.output = nn.Linear(hidden_dim, vocab_size)

  def forward(self, pair):
    src_embeddings = self.embed(pair.src_token_ids)
    tgt_embeddings = self.embed(pair.tgt_token_ids)
    src_lengths = pair.src_mask.sum(dim=1)
    src_packed = pack_padded_sequence(
        src_embeddings, lengths=src_lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, hidden_states=self.encoder(src_packed)
    outputs, _ = self.decoder(tgt_embeddings, hidden_states)
    return self.output(outputs).permute(0,2,1)

torch.manual_seed(69)
vocab_size = nmt_tokenizer.get_vocab_size()
nmt_model=NmtModel(vocab_size).to(device)

# run the training

n_epochs = 10
xentropy = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad> tokens
optimizer = torch.optim.NAdam(nmt_model.parameters(), lr=0.001)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size)
accuracy = accuracy.to(device)

history = train(nmt_model, optimizer, xentropy, accuracy,
                nmt_train_loader, nmt_valid_loader, n_epochs)

# testing evaluation

"""# Deployment"""

torch.save(nmt_model.state_dict(), "my_nmt_model.pt")

# inference
def translate(model, src_text, max_length=20, pad_id=0, eos_id=3):
    tgt_text = ""
    token_ids = []
    for index in range(max_length):
        batch, _ = nmt_collate_fn([{"source_text": src_text,
                                    "target_text": tgt_text}])
        with torch.no_grad():
            Y_logits = model(batch.to(device))
            Y_token_ids = Y_logits.argmax(dim=1)  # find the best token IDs
            next_token_id = Y_token_ids[0, index]  # take the last token ID

        next_token = nmt_tokenizer.id_to_token(next_token_id)
        tgt_text += " " + next_token
        if next_token_id == eos_id:
            break
    return tgt_text

nmt_model.eval()
translate(nmt_model, "I dare you, I double dare you motherfucker")

nmt_model.eval()
translate(nmt_model, "I dare you motherfucker")

nmt_model.eval()
translate(nmt_model, "I went to the forest and saw three elephants")

nmt_model.eval()
translate(nmt_model, "I like elephants")

longer_text = "I like to play soccer with my friends."
translate(nmt_model, longer_text)

"""# Beam Search"""

# implementation of simple beam search algorithm:

def beam_search(model, src_text, beam_width=3, max_length=20,
                verbose=False, length_penalty=0.6):
    top_translations = [(torch.tensor(0.), "")]
    for index in range(max_length):
        if verbose:
            print(f"Top {beam_width} translations so far:")
            for log_proba, tgt_text in top_translations:
                print(f"    {log_proba.item():.3f} – {tgt_text}")

        candidates = []
        for log_proba, tgt_text in top_translations:
            if tgt_text.endswith(" </s>"):
                candidates.append((log_proba, tgt_text))
                continue  # don't add tokens after EOS token
            batch, _ = nmt_collate_fn([{"source_text": src_text,
                                        "target_text": tgt_text}])
            with torch.no_grad():
                Y_logits = model(batch.to(device))
                Y_log_proba = F.log_softmax(Y_logits, dim=1)
                Y_top_log_probas = torch.topk(Y_log_proba, k=beam_width, dim=1)

            for beam_index in range(beam_width):
                next_token_log_proba = Y_top_log_probas.values[0, beam_index, index]
                next_token_id = Y_top_log_probas.indices[0, beam_index, index]
                next_token = nmt_tokenizer.id_to_token(next_token_id)
                next_tgt_text = tgt_text + " " + next_token
                candidates.append((log_proba + next_token_log_proba, next_tgt_text))

        def length_penalized_score(candidate, alpha=length_penalty):
            log_proba, text = candidate
            length = len(text.split())
            penalty = ((5 + length) ** alpha) / (6 ** alpha)
            return log_proba / penalty

        top_translations = sorted(candidates,
                                  key=length_penalized_score,
                                  reverse=True)[:beam_width]

    return top_translations[-1][1]

import torch.nn.functional as F

beam_search(nmt_model, longer_text, beam_width=3)

beam_search(nmt_model, "I dare you motherfucker", beam_width=3)

beam_search(nmt_model, "I saw three elephants", beam_width=3)

longest_text = "I like to play soccer with my friends at the beach."
beam_search(nmt_model, longest_text, beam_width=3)

"""# Attention Mechanism"""

# free-up gpu ram
del_vars(["nmt_model"])

def attention(query, key, value): # note: dq == dk and Lk == Lv
  scores = query @ key.transpose(1, 2) # [B,Lq,dq] @ [B,dk,Lk] = [B, Lq, Lk]
  weights = torch.softmax(scores, dim=-1) # [B, Lq, Lk]
  return weights @ value # [B, Lq, Lk] @ [B, Lv, dv] = [B, Lq, dv]

class NmtModelWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, pad_id=0, hidden_dim=512,
                 n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(
            embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.decoder = nn.GRU(
            embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.output = nn.Linear(2 * hidden_dim, vocab_size)

    def forward(self, pair):
        src_embeddings = self.embed(pair.src_token_ids)  # same as earlier
        tgt_embeddings = self.embed(pair.tgt_token_ids)  # same
        src_lengths = pair.src_mask.sum(dim=1)  # same
        src_packed = pack_padded_sequence(
            src_embeddings, lengths=src_lengths.cpu(),
            batch_first=True, enforce_sorted=False)  # same
        encoder_outputs_packed, hidden_states = self.encoder(src_packed)
        decoder_outputs, _ = self.decoder(tgt_embeddings, hidden_states)  # same
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed,
                                                 batch_first=True)
        attn_output = attention(query=decoder_outputs,
                                key=encoder_outputs, value=encoder_outputs)
        combined_output = torch.cat((attn_output, decoder_outputs), dim=-1)
        return self.output(combined_output).permute(0, 2, 1)

torch.manual_seed(42)
nmt_attn_model = NmtModelWithAttention(vocab_size).to(device)

n_epochs = 10
xentropy = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad> tokens
optimizer = torch.optim.NAdam(nmt_attn_model.parameters(), lr=0.001)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size)
accuracy = accuracy.to(device)

history = train(nmt_attn_model, optimizer, xentropy, accuracy,
                nmt_train_loader, nmt_valid_loader, n_epochs)

torch.save(nmt_attn_model.state_dict(), "my_nmt_attn_model.pt")

translate(nmt_attn_model, longer_text)
