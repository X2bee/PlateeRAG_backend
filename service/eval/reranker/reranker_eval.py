from sentence_transformers import CrossEncoder
from datasets import load_dataset
import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def eval_reranker(model, eval_dataset, at_k=10):

    samples = [
        {
            "query": sample["query"],
            "positive": [text for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"]) if is_selected],
            "documents": sample["passages"]["passage_text"],
            # or
            # "negative": [text for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"]) if not is_selected],
        }
        for sample in eval_dataset
    ]
    
    logging.info("샘플 추출 완료: 총 %d개의 샘플", len(samples))
    
    all_mrr_scores = []
    all_ndcg_scores = []
    all_ap_scores = []
    
    for sample in tqdm(samples, desc="Evaluating reranker"):
        query = sample["query"]
        positive = sample["positive"]
        docs = sample["documents"]
        
        # 각 문서가 positive에 포함되어 있으면 1, 아니면 0
        is_relevant = [1 if doc in positive else 0 for doc in docs]
        
        # 만약 positive 문서가 하나도 없으면 평가하지 않습니다.
        if sum(is_relevant) == 0:
            all_mrr_scores.append(0)
            all_ndcg_scores.append(0)
            all_ap_scores.append(0)
            continue
        
        # 모델 입력 생성: [query, 문서] 쌍
        model_input = [[query, doc] for doc in docs]
        pred_scores = model.predict(model_input, convert_to_numpy=True)
        
        # MRR@at_k 계산: 상위 at_k개의 순위에서 처음으로 positive 문서를 찾았을 때의 reciprocal rank
        ranking = np.argsort(pred_scores)[::-1]
        mrr = 0
        for rank, idx in enumerate(ranking[:at_k]):
            if is_relevant[idx]:
                mrr = 1 / (rank + 1)
                break
        
        # NDCG@at_k와 MAP 계산
        ndcg = ndcg_score([is_relevant], [pred_scores], k=at_k)
        ap = average_precision_score(is_relevant, pred_scores)
        
        all_mrr_scores.append(mrr)
        all_ndcg_scores.append(ndcg)
        all_ap_scores.append(ap)
    
    mean_mrr = np.mean(all_mrr_scores)
    mean_ndcg = np.mean(all_ndcg_scores)
    mean_ap = np.mean(all_ap_scores)
    
    logging.info("평가 결과: MRR@%d: %f, NDCG@%d: %f, AP: %f", at_k, mean_mrr, at_k, mean_ndcg, mean_ap)
    
    # model_card_metrics 딕셔너리: 숫자형 값으로 저장합니다.
    model_card_metrics = {
        "map": mean_ap,
        f"mrr@{at_k}": mean_mrr,
        f"ndcg@{at_k}": mean_ndcg,
    }
    
    return model_card_metrics