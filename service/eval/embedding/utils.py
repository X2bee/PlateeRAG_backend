from torch.utils.data import Dataset
import torch 
import math 

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data  # data는 list of dicts 형태여야 합니다.
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        # 값 중 하나라도 문자열이 있으면, 그대로 리스트로 수집
        if any(isinstance(v, str) for v in values):
            collated[key] = values
        else:
            collated[key] = torch.tensor(values, dtype=torch.float64)
    return collated

def compute_dcg(relevances):
    """주어진 relevance 리스트로 DCG를 계산합니다."""
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += (2**rel - 1) / math.log2(i + 2)  # i+2: 1-based index에 대응
    return dcg

def compute_ndcg(sim_scores, true_labels):
    """
    sim_scores: 리스트 형태의 유사도 값 (예, [positive_sim, negative_sim])
    true_labels: 해당 항목들의 relevance 라벨 (positive: 1, negative: 0)
    """
    # 유사도 기준 내림차순 정렬한 인덱스
    sorted_indices = sorted(range(len(sim_scores)), key=lambda x: sim_scores[x], reverse=True)
    dcg = 0.0
    for i, idx in enumerate(sorted_indices):
        rel = true_labels[idx]
        dcg += (2**rel - 1) / math.log2(i + 2)  # 순위 i+1 (인덱스 i -> rank = i+1)
        
    # 이상적인 ranking: relevance가 높은 순으로 정렬
    sorted_true_labels = sorted(true_labels, reverse=True)
    ideal_dcg = 0.0
    for i, rel in enumerate(sorted_true_labels):
        ideal_dcg += (2**rel - 1) / math.log2(i + 2)
        
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0