import os
import torch
from PIL import Image
from torchvision import transforms
from clip import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

class_names = ["Beagle", "Boxer", "Bulldog", "Dachshund"]
class_descriptions = [
    "a photo of a beagle dog",
    "a photo of a boxer dog",
    "a photo of a bulldog dog",
    "a photo of a dachshund dog"
]

text_inputs = torch.cat([clip.tokenize(desc) for desc in class_descriptions]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

test_image_dir = r"d:\GZU\badou\作业\十\dataset"
test_images = []
true_labels = []
image_paths = []

for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(test_image_dir, class_name)
    if os.path.exists(class_dir):
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                image = Image.open(img_path).convert("RGB")
                test_images.append(preprocess(image))
                true_labels.append(class_idx)
                image_paths.append(img_path)

test_images = torch.stack(test_images).to(device)

with torch.no_grad():
    image_features = model.encode_image(test_images)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    logits_per_image = 100.0 * image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1)
    preds = probs.argmax(dim=-1)

correct = 0
for i, (pred, true) in enumerate(zip(preds, true_labels)):
    is_correct = pred.item() == true
    correct += is_correct
    print(f"{image_paths[i]}: 预测为 {class_names[pred.item()]} (真实: {class_names[true]}), {'正确' if is_correct else '错误'}")

accuracy = correct / len(true_labels)
print(f"\n准确率: {accuracy:.2%} ({correct}/{len(true_labels)})")
