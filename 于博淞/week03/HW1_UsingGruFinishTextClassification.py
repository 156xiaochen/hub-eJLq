import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
num_classes = len(label_to_index)
max_len = 40

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels
)

class CharSeqDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

class CharRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='LSTM'):
        super(CharRNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        rnn_types = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        self.rnn = rnn_types[rnn_type](embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.rnn_type = rnn_type

    def forward(self, x):
        embedded = self.embedding(x)
        if self.rnn_type == 'LSTM':
            rnn_out, (hidden, _) = self.rnn(embedded)
        else:
            rnn_out, hidden = self.rnn(embedded)
        out = self.fc(hidden.squeeze(0))
        return out


def train_model(model, train_loader, val_loader, num_epochs=4, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        correct_train, total_train = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train

        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f"[{model.rnn_type}] Epoch {epoch + 1}/{num_epochs} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    return max(val_acc_history)  # 返回最佳验证精度


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embedding_dim = 64
    hidden_dim = 128
    batch_size = 32
    num_epochs = 4
    lr = 0.001

    train_dataset = CharSeqDataset(train_texts, train_labels, char_to_index, max_len)
    val_dataset = CharSeqDataset(val_texts, val_labels, char_to_index, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    results = {}

    for rnn_type in ['RNN', 'LSTM', 'GRU']:
        print(f"\n{'=' * 50}")
        print(f"Training {rnn_type} model...")
        print(f"{'=' * 50}")

        model = CharRNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            rnn_type=rnn_type
        )

        best_acc = train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs, lr=lr, device=device
        )
        results[rnn_type] = best_acc

    # 输出对比结果
    print("\n" + "=" * 60)
    print("FINAL COMPARISON (Best Validation Accuracy):")
    print("=" * 60)
    for model_name, acc in results.items():
        print(f"{model_name:5} : {acc:.2f}%")
    print("=" * 60)