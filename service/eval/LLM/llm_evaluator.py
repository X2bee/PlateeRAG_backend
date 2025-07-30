import torch
from tqdm.auto import tqdm
import logging

# tqdm 출력 메시지를 logger로 전달하기 위한 클래스
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

def evaluate_causal_lm(
    model,
    tokenizer,
    ds,
    gpu_count: int = 1,
    max_new_tokens: int = 16,
    use_cot: bool = False,
    question_field:   str = "question",
    options_field:    str = "options",
    cot_field:        str = "cot_content",
    answer_idx_field: str = "answer_index",
    category_field:   str = "category",
):
    # 1) 데이터 로드
    # (데이터셋 ds는 외부에서 이미 로드되어 전달된 것으로 가정)

    # 2) 사용할 split 자동 선택 (print 대신 logging 사용)
    for split in ("test", "validation", "train"):
        if split in ds:
            data = ds[split]
            logging.info(f">>> using split '{split}' ({len(data)} examples)")
            break
    else:
        data = ds

    # 3) 모델 & 토크나이저 로드
    # (모델과 토크나이저는 이미 로드되어 있다고 가정)

    has_category = any(category_field in ex and ex[category_field] for ex in data)
    if has_category:
        per_cat = {}   # { category_value: { "correct": int, "total": int } }
    total_correct = 0
    total_count   = 0

    # 4) GPU 세팅
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_count > 1 and torch.cuda.device_count() >= gpu_count:
        model = torch.nn.DataParallel(model, device_ids=list(range(gpu_count)))
    model.to(device)
    model.eval()

    # 5) 통계 변수 초기화 (has_category가 있을 경우 per_cat 재설정)
    if has_category:
        per_cat = {}
    total_correct = 0
    total_count = 0

    # 6) 평가 루프 (tqdm 진행 표시를 TqdmToLogger를 이용해 logging으로 처리)
    for ex in tqdm(data, desc="Evaluating CausalLM", file=TqdmToLogger(logging.getLogger(__name__))):
        question     = ex[question_field].strip()
        options      = ex[options_field]
        answer_index = ex[answer_idx_field]
        cot_example  = ex.get(cot_field, "").strip()

        if has_category:
            category = ex.get(category_field, "default")
            per_cat.setdefault(category, {"correct": 0, "total": 0})

        # (a) 프롬프트 구성
        prompt = ""
        if use_cot and cot_example:
            prompt += cot_example + "\n\n"
        prompt += f"Question: {question}\nOptions:\n"
        for idx, opt in enumerate(options):
            prompt += f"{idx}. {opt}\n"
        prompt += "Answer:"

        # (b) 토크나이즈 & 생성
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        gen = tokenizer.decode(
            out_ids[0][ inputs["input_ids"].shape[-1] : ],
            skip_special_tokens=True
        ).strip()

        # (c) 예측 인덱스 파싱
        pred_idx = -1
        for tok in gen.replace("\n", " ").split():
            tok = tok.rstrip(".:")
            if tok.isdigit() and 0 <= int(tok) < len(options):
                pred_idx = int(tok)
                break

        # (d) 정답 비교 및 집계
        if pred_idx == answer_index:
            total_correct += 1
            if has_category:
                per_cat[category]["correct"] += 1
        total_count += 1
        if has_category:
            per_cat[category]["total"] += 1

    # 7) 결과 계산
    overall_acc = total_correct / total_count * 100
    result = {
        "overall_accuracy": overall_acc,
        "total_examples": total_count
    }
    if has_category:
        per_cat_acc = {
            cat: stats["correct"] / stats["total"] * 100
            for cat, stats in per_cat.items()
        }
        result["per_category_accuracy"] = per_cat_acc

    return result