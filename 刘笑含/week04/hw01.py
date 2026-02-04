import numpy as np
import pandas as pd

import torch
from torch.utils.data  import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from evaluate import load as load_metric

dataset = load_dataset("FreedomIntelligence/Huatuo26M-Lite")



n_samples = 10000
questions = dataset['train']['question'][:n_samples]
labels = dataset['train']['label'][:n_samples]

label_encoder = LabelEncoder()
label_encoder.fit(labels)

x_train, x_test, y_train, y_test = train_test_split(questions, labels, test_size=0.2, random_state=42, stratify=labels)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)


tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-chinese', num_labels=len(label_encoder.classes_))

max_length = 100
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=max_length, return_token_type_ids=False)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=max_length, return_token_type_ids=False)


class ModelDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['token_type_ids'] = torch.zeros_like(item['input_ids'], dtype=torch.long)
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = ModelDataset(train_encodings, y_train)
test_dataset = ModelDataset(test_encodings, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
accuracy_metric = load_metric("accuracy")


def train_model(model, optimizer, train_loader, num_epochs, device):
    model.train()
    n_batches = len(train_loader)

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            print('train batch', batch_idx)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'], token_type_ids=batch['token_type_ids'], labels=batch['labels'])
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1} batch {batch_idx + 1}/{n_batches}, loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / n_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_train_loss:.4f}")
    return avg_train_loss


def evaluate_model(model, data_loader, device):
    model.eval()
    total_eval_accuracy = 0.0
    total_eval_loss = 0.0

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch['input_ids'], batch['attention_mask'], token_type_ids=batch['token_type_ids'], labels=batch['labels'])
        loss = outputs.loss
        total_eval_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        total_eval_accuracy += accuracy_metric.compute(predictions=preds.cpu().numpy(), references=batch['labels'].cpu().numpy())['accuracy']
    avg_eval_loss = total_eval_loss / len(data_loader)
    avg_eval_accuracy = total_eval_accuracy / len(data_loader)
    print(f"Evaluation Loss: {avg_eval_loss:.4f}, Evaluation Accuracy: {avg_eval_accuracy:.4f}")



print("Training...")
num_epochs = 3
train_model(model, optimizer, train_loader, num_epochs, device)
print("Evaluating...")
evaluate_model(model, test_loader, device)


