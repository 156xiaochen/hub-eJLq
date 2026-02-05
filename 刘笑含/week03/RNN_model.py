import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


dataset = pd.read_csv("../../data/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label:i for i, label in enumerate(set(string_labels))}
index_to_label = {i:label for label,i in label_to_index.items()}
encoded_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
  for char in text:
    if char not in char_to_index:
      char_to_index[char] = len(char_to_index)

index_to_char = {i:char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40

class ModelDataset(Dataset):
  def __init__(self, texts, labels, char_to_index, max_len):
    self.texts = texts
    self.labels = torch.tensor(labels, dtype = torch.long)
    self.char_to_index = char_to_index
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)


  def __getitem__(self,index):
    text = self.texts[index]
    indices = [self.char_to_index.get(char,0) for char in text[:self.max_len]]
    indices += [0] * (self.max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long), self.labels[index]




class RNNClassifier(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
    super(RNNClassifier, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first = True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    embedded = self.embedding(x)
    rnn_out, hidden_state = self.rnn(embedded)
    out = self.fc(hidden_state.squeeze(0))
    return out





model_dataset = ModelDataset(texts, encoded_labels, char_to_index, max_len)
dataloader = DataLoader(model_dataset, batch_size = 32, shuffle = True)


embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr = 0.001)


num_epochs = 12
loss_history = []

for epoch in range(num_epochs):
  rnn_model.train()
  running_loss_rnn = 0.0
  for index, (input, labels) in enumerate(dataloader):
    optimizer_rnn.zero_grad()
    rnn_outputs = rnn_model(input)
    loss_rnn = criterion(rnn_outputs, labels)
    loss_rnn.backward()
    torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), max_norm=1.0)
    optimizer_rnn.step()
    running_loss_rnn += loss_rnn.item()
    if index % 50 == 0:
      print(f"Batch 个数 {index}, 当前Batch Loss_rnn: {loss_rnn.item():.4f}")
  epoch_loss = running_loss_rnn / len(dataloader)
  loss_history.append(epoch_loss)
  print(f"Epoch [{epoch + 1}/{num_epochs}], Loss_rnn: {epoch_loss:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(range(1, num_epochs + 1), loss_history, "o-", color="coral")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RNN Loss Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rnn_loss.png", dpi=120)
plt.show()



def classify_text(text, model, char_to_index, max_len, index_to_label):
  indices = [char_to_index.get(char, 0) for char in text[:max_len]]
  indices += [0] * (max_len - len(indices))
  input_tensor = torch.tensor(indices, dtype =torch.long).unsqueeze(0)

  model.eval()
  with torch.no_grad():
    output = model(input_tensor)

  _, predicted_index = torch.max(output,1)
  predicted_index = predicted_index.item()
  predicted_label = index_to_label[predicted_index]
  return predicted_label



new_text = "帮我导航到北京"
predicted_class_rnn = classify_text(new_text, rnn_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class_rnn}'")

new_text_2 = "查询明天北京的天气"
predicted_class_rnn_2 = classify_text(new_text_2, rnn_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_rnn_2}'")


