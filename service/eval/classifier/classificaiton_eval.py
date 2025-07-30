import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
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

def compute_classfication(model, tokenizer, dataset, label, text, batch_size=32, device='cuda', top_k=1, num_gpus=1):
    """
    모델과 토크나이저, 데이터셋을 받아 top-k accuracy를 계산하는 함수입니다.
    텍스트 라벨과 숫자 라벨 모두를 지원하기 위해, 텍스트 라벨인 경우 숫자 인덱스로 매핑합니다.
    
    Args:
        model: 이미 로드된 모델 (예: BertForSequenceClassification)
        tokenizer: 이미 로드된 토크나이저 (예: AutoTokenizer 또는 BertTokenizerFast)
        dataset: 평가할 데이터셋 (각 샘플에 'text'와 'label' 키가 있다고 가정)
        batch_size: 배치 사이즈
        device: 모델이 위치한 디바이스 (예: 'cuda' 또는 'cpu')
        top_k: top-k accuracy를 계산할 k 값 (기본값: 1)
        num_gpus: 사용하고자 하는 GPU 개수 (기본값: 1)
    
    Returns:
        전체 데이터셋에 대한 top-k accuracy (0과 1 사이의 값)를 담은 dict
    """
    model.eval()
    model.to(device)
    
    # GPU 설정
    if device == 'cuda' and torch.cuda.device_count() > 1:
        available_gpus = torch.cuda.device_count()
        use_gpus = min(num_gpus, available_gpus)
        if use_gpus > 1:
            device_ids = list(range(use_gpus))
            model = nn.DataParallel(model, device_ids=device_ids)
            logging.info(f"Using {use_gpus} GPUs out of {available_gpus} available.")
        else:
            logging.info("Using a single GPU.")
    else:
        logging.info("Using CPU or a single GPU.")
    
    # DataLoader 생성
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # 라벨이 텍스트인지 숫자인지 판별 및 텍스트인 경우 매핑 딕셔너리 생성
    sample = dataset[0]  # 첫 샘플을 확인 (데이터셋이 indexable하다는 가정 하에)
    if isinstance(sample[label], str):
        unique_labels = list(set([sample[label] for sample in dataset]))
        label_to_id = {l: i for i, l in enumerate(unique_labels)}
        logging.info("Label mapping: %s", label_to_id)
    else:
        label_to_id = None
    
    total_correct = 0
    total_samples = 0
    
    # tqdm 진행 표시 시 TqdmToLogger를 통해 logging 처리
    for batch in tqdm(dataloader, desc="Evaluating", file=TqdmToLogger(logging.getLogger(__name__))):
        texts = batch[text]
        batch_labels = batch[label]
        
        # 텍스트 라벨인 경우, 숫자로 변환
        if label_to_id is not None:
            labels = torch.tensor([label_to_id[l] for l in batch_labels]).to(device)
        else:
            labels = batch_labels.to(device)
        
        # 토큰화 및 모델 실행
        encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encodings = {key: value.to(device) for key, value in encodings.items()}
        
        with torch.no_grad():
            outputs = model(**encodings)
        logits = outputs.logits  # (batch_size, num_labels)
        
        # Top-k 예측 비교
        if top_k == 1:
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
        else:
            topk_preds = torch.topk(logits, k=top_k, dim=1).indices
            correct = topk_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            total_correct += correct
        total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    return {'accuracy': accuracy}