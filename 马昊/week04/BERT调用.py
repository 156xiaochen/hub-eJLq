from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib
import os

# -------------------------- 1. 初始化FastAPI应用 --------------------------
app = FastAPI(
    title="银行客服文本分类API",
    description="基于BERT的银行客服对话文本分类接口（仅推理）",
    version="1.0.0"
)

# -------------------------- 2. 配置路径（替换成你的实际路径！） --------------------------
MODEL_PATH = r"C:\Users\48628\Desktop\text_classification_model_results\final_model"
TOKENIZER_PATH = r"C:\Users\48628\Desktop\text_classification_model_results\final_tokenizer"
LABEL_ENCODER_PATH = r"C:\Users\48628\Desktop\text_classification_model_results\label_encoder.pkl"

# -------------------------- 3. 加载组件（增加异常捕获） --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 加载分词器
try:
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    print("分词器加载成功 ✅")
except Exception as e:
    print(f"分词器加载失败：{e}")
    exit(0)

# 加载模型
try:
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
    print("模型加载成功 ✅")
except Exception as e:
    print(f"模型加载失败：{e}")
    exit(0)

# 加载标签编码器
try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("标签编码器加载成功 ✅")
except Exception as e:
    print(f"标签编码器加载失败：{e}")
    exit(0)


# -------------------------- 4. Pydantic V2兼容写法（无警告） --------------------------
class TextClassificationInput(BaseModel):
    text: str = Field(
        ...,
        description="需要分类的银行客服文本内容",
        json_schema_extra={"example": "我的银行卡丢了，该怎么挂失？"}  # V2正确写法
    )


class TextClassificationOutput(BaseModel):
    text: str = Field(description="输入的原始文本")
    label: str = Field(description="分类结果（文本标签）")
    label_id: int = Field(description="分类结果的数字标签")
    confidence: float = Field(description="分类置信度（0-1）")


# -------------------------- 5. 核心推理函数 --------------------------
def predict_text(text: str) -> dict:
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)
    confidence, label_id = torch.max(probabilities, dim=-1)

    label_id = label_id.item()
    confidence = round(confidence.item(), 4)
    label = label_encoder.inverse_transform([label_id])[0]

    return {
        "text": text,
        "label": label,
        "label_id": label_id,
        "confidence": confidence
    }


# -------------------------- 6. API接口 --------------------------
@app.post("/predict", response_model=TextClassificationOutput, summary="文本分类预测")
async def predict(input_data: TextClassificationInput):
    try:
        result = predict_text(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")


@app.get("/health", summary="健康检查")
async def health_check():
    return {
        "status": "success",
        "message": "API服务运行正常",
        "model_loaded": True,
        "device": str(device)
    }


# -------------------------- 7. 稳定启动服务（无reload，无警告） --------------------------
if __name__ == "__main__":
    import uvicorn

    print("\n===== 启动API服务（仅推理，不训练） =====")
    # 核心修复：关闭reload，用127.0.0.1替代0.0.0.0，避免所有警告
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，最稳定
        port=8000,
        reload=False,  # 关闭reload，彻底消除警告
        log_level="info"  # 显示详细启动日志
    )
