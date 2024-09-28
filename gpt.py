import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def encode_dataset(examples):
    return tokenizer(examples['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(encode_dataset, batched=True)
dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_head), n_layer
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        transformer_out = self.transformer_layers(embedded)
        logits = self.fc_out(transformer_out)
        return logits

d_model = 128
n_head = 4
n_layer = 4
vocab_size = tokenizer.vocab_size

model = GPT(vocab_size, d_model, n_head, n_layer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 2

for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids']
        labels = input_ids.clone()
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
        
        loss.backward()
        optimizer.step()
    
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

print("Training complete!")
