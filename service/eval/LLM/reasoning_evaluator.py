from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader

# 1) 설정
model_name = "gpt2"  # 원하는 causal LM 이름
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# 2) 데이터셋 로드
ds = load_dataset("hellaswag", "default", split="validation")

# 3) 배치 단위 처리 함수
def collate_fn(batch):
    examples = []
    for ex in batch:
        ctx = ex["ctx"]
        endings = ex["endings"]  # 문자열 리스트 형태의 문자열
        # 파이썬 리스트로 변환
        candidates = eval(endings)
        examples.append({
            "ctx": ctx,
            "candidates": candidates,
            "label": int(ex["label"])
        })
    return examples

loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)

# 4) 평가
correct = 0
total = 0

for batch in loader:
    for ex in batch:
        ctx = ex["ctx"]
        candidates = ex["candidates"]
        label = ex["label"]

        scores = []
        # 각 candidate에 대해 score 계산
        for cand in candidates:
            text = ctx + cand
            inputs = tokenizer(text, return_tensors="pt").to(device)

            # context 길이만큼 토큰 마스킹
            ctx_len = len(tokenizer(ctx)["input_ids"])
            labels = inputs["input_ids"].clone()
            labels[:, :ctx_len] = -100  # context 토큰은 로스 계산에서 제외

            with torch.no_grad():
                out = model(**inputs, labels=labels)
                # out.loss 은 batch 평균 loss (여기선 batch size=1)
                scores.append(-out.loss.item())

        pred = int(torch.argmax(torch.tensor(scores)))
        if pred == label:
            correct += 1
        total += 1

accuracy = correct / total
print(f"HellaSwag validation accuracy: {accuracy:.4f}")