import json
from sacrebleu import corpus_bleu
from tqdm import tqdm
import torch
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sacrebleu import corpus_bleu
from tqdm import tqdm
"""
# 1) 모델·토크나이저 로드
model_name = 'naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
model.eval()

# 2) 데이터셋 로드
ds = load_dataset('beomi/KoAlpaca-v1.1a', split='train')
ds = ds.select(range(1000))

# 3) 프롬프트 생성 및 응답 함수
def make_prompt(sample):
    instr = sample['instruction'].strip()
    inp   = sample.get('input', '').strip()
    if inp:
        return f"{instr}\nInput: {inp}\nResponse:"
    else:
        return f"{instr}\nResponse:"

@torch.no_grad()
def generate(prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # 프롬프트 뒤에 붙은 생성문장만 떼어내기
    return text[len(prompt):].strip()

# 4) 참조·후보 수집
refs, cands = [], []
for sample in tqdm(ds, desc="Generating"):
    prompt = make_prompt(sample)
    pred   = generate(prompt)
    # KoAlpaca 필드는 "output"에 레퍼런스 답변이 있음
    refs.append(sample['output'].strip())
    cands.append(pred)

# 5) BLEU 계산
bleu = corpus_bleu(cands, [refs], force=True)
print(f"Corpus BLEU score: {bleu.score:.2f}")
"""

# 1) 데이터셋 로드
class TqdmToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
    def write(self, message):
        message = message.strip()
        if message:
            self.logger.log(self.level, message)
    def flush(self):
        pass

def make_prompt(sample, instruction, input):
    instr = sample[instruction].strip()
    inp   = sample.get(input, '').strip()
    if inp:
        return f"{instr}\nInput: {inp}\nResponse:"
    else:
        return f"{instr}\nResponse:"

@torch.no_grad()
def generate(prompt, model, tokenizer, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

def evaluate_bleu(
    model,
    tokenizer,
    ds,
    gpu_count: int = 1,
    max_new_tokens: int = 256,
    input:   str = "input",
    instruction:    str = "instruction",
    output:        str = "output",
):
    refs, cands = [], []
    for sample in tqdm(ds, desc="Generating", file=TqdmToLogger(logging.getLogger(__name__))):
        prompt = make_prompt(sample, input, instruction)
        pred   = generate(prompt, model, tokenizer, max_new_tokens)
        refs.append(sample[output].strip())
        cands.append(pred)
    bleu = corpus_bleu(cands, [refs], force=True)
    return {'bleu' : bleu} 