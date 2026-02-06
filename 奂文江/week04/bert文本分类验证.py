from typing import Union, List
from fastapi import FastAPI
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

app = FastAPI()

from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification

CATEGORY_NAME = ["交通运输仓储邮政", "住宿餐饮", "信息软件",  "农业",  "制造业", "卫生医疗", "国际组织", "建筑", "房地产",
                 "政府组织", "教育", "文体娱乐", "水利环境", "电力燃气水生产", "科学技术", "租赁法律", "采矿", "金融"
                 ]
# CATEGORY_NAME = ["冷藏即饮果汁","即饮奶茶","即饮茶","口香糖","方便面","牙膏","白酒","硬糖","软糖","非冷藏即饮果汁"]

BERT_MODEL_PKL_PATH = "./results/bert.pt"
BERT_MODEL_PERTRAINED_PATH = "../models/google-bert/bert-base-chinese/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PERTRAINED_PATH)
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PERTRAINED_PATH, num_labels=10)

model.load_state_dict(torch.load(BERT_MODEL_PKL_PATH))
model.to(device)


class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

@app.post("/v1/text-cls/bert")
def model_for_bert(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    classify_result: Union[str, List[str]] = None

    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("格式不支持")
  #fastapi启动时打的断点
  #  import pdb; pdb.set_trace()

    test_encoding = tokenizer(list(request_text), truncation=True, padding=True, max_length=64)
    test_dataset = NewsDataset(test_encoding, [0] * len(request_text))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()
    pred = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        pred += list(np.argmax(logits, axis=1).flatten())

    classify_result = [CATEGORY_NAME[x] for x in pred]
    return classify_result
