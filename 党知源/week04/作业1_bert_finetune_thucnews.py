"""
作业1: BERT文本分类微调,使用的开源数据集是THUCNews
"""

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
import pickle

def load_thucnews_dataset():
    try:
        dataset = load_dataset("SirlyDreamer/THUCNews")
        df = pd.DataFrame(dataset['train'])
        print(f"数据集加载成功: {len(df)} 条")
        return df
    except Exception as e:
        print(f"加载失败: {e}")
        return None

def preprocess_data(df):
    df['content'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    texts = df['content'].astype(str).tolist()
    labels = df['label'].astype(str).tolist()
    return texts, labels

def train_bert(train_texts, train_labels, test_texts, test_labels, model_name='bert-base-chinese', output_dir='./bert_thucnews'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    all_labels = list(set(train_labels + test_labels))
    num_labels = len(all_labels)
    
    label_encoder = LabelEncoder()
    train_encoded_labels = label_encoder.fit_transform(train_labels)
    test_encoded_labels = label_encoder.transform(test_labels)
    
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64, return_tensors='pt')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64, return_tensors='pt')
    
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'].tolist(),
        'attention_mask': train_encodings['attention_mask'].tolist(),
        'labels': train_encoded_labels.tolist()
    })
    
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'].tolist(),
        'attention_mask': test_encodings['attention_mask'].tolist(),
        'labels': test_encoded_labels.tolist()
    })
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {'accuracy': (predictions == labels).mean()}
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        report_to=None,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("开始训练...")
    trainer.train()
    
    final_model_path = f"{output_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    with open(f"{final_model_path}/label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return trainer, tokenizer, label_encoder, final_model_path

def test_model(model_path, tokenizer, label_encoder, test_samples):
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    for i, text in enumerate(test_samples, 1):
        encoding = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_id = torch.argmax(outputs.logits, dim=-1).item()
            confidence = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][predicted_id].item()
        
        predicted_label = label_encoder.inverse_transform([predicted_id])[0]
        print(f"样本{i}: {text[:50]}... -> {predicted_label} ({confidence:.4f})")

def main():
    df = load_thucnews_dataset()
    if df is None:
        return
    
    texts, labels = preprocess_data(df)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    train_df_sampled = train_df.sample(n=min(1000, len(train_df)), random_state=42)
    train_texts = train_df_sampled['text'].tolist()
    train_labels = train_df_sampled['label'].tolist()

    test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
    test_df_sampled = test_df.sample(n=min(100, len(test_df)), random_state=42)
    test_texts = test_df_sampled['text'].tolist()
    test_labels = test_df_sampled['label'].tolist()
    
    print(f"训练集: {len(train_texts)} 条")
    print(f"测试集: {len(test_texts)} 条")
    
    trainer, tokenizer, label_encoder, model_path = train_bert(
        train_texts, train_labels, test_texts, test_labels
    )
    
    if model_path is None:
        return
    
    test_samples = [
        "中国队在奥运会上获得金牌，创造了历史最好成绩",
        "科技公司发布最新人工智能产品，引发市场关注"
    ]
    
    test_model(model_path, tokenizer, label_encoder, test_samples)
    print(f"\n模型保存路径: {model_path}")

if __name__ == '__main__':
    main()
