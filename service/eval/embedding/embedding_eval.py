import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
from service.eval.embedding.utils import CustomDataset, custom_collate_fn, compute_dcg, compute_ndcg
import logging
logging.basicConfig(level=logging.INFO)

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

# ---------------------------
def extract_text(example, positive_field, negative_field):
    pos_val = example.get(positive_field)
    if isinstance(pos_val, list) and pos_val:
        if isinstance(pos_val[0], dict):
            example[positive_field] = pos_val[0].get('text', '')
        elif isinstance(pos_val[0], str):
            # 이미 문자열이면 그대로 사용
            example[positive_field] = pos_val[0]
        else:
            example[positive_field] = str(pos_val[0])
    else:
        example[positive_field] = pos_val

    neg_val = example.get(negative_field)
    if isinstance(neg_val, list) and neg_val:
        if isinstance(neg_val[0], dict):
            example[negative_field] = neg_val[0].get('text', '')
        elif isinstance(neg_val[0], str):
            example[negative_field] = neg_val[0]
        else:
            example[negative_field] = str(neg_val[0])
    else:
        example[negative_field] = neg_val

    logging.info("변환 후 positive_field: %s", example.get(positive_field))
    return example

# ---------------------------
def evaluate_retriever_np_ndvg(model, tokenizer, datasets, query_field, positive_field, negative_field, device='cuda', batch_size=32, gpu_num=1):
    """
    모델과 토크나이저, 데이터를 배치 처리 및 GPU 개수를 설정하여 
    query, positive, negative 텍스트의 임베딩으로 코사인 유사도를 기반으로 NDCG 평가 지표를 계산하는 함수입니다.
    
    인자:
      - model: Transformer 기반의 임베딩 모델
      - tokenizer: 해당 모델에 맞는 토크나이저
      - datasets: 평가할 데이터셋
          * 만약 dict (컬럼 단위의 데이터) 형태라면 자동으로 리스트(각 샘플이 dict) 형태로 변환합니다.
      - query_field: query 텍스트를 담고 있는 키의 이름 (예: "query")
      - positive_field: positive 텍스트를 담고 있는 키의 이름 (예: "positive")
      - negative_field: negative 텍스트를 담고 있는 키의 이름 (예: "negative")
      - device: 모델을 올릴 디바이스 (기본 'cuda')
      - batch_size: 배치 사이즈 (기본값 32)
      - gpu_num: 사용할 GPU 개수 (기본값 1)
    
    반환:
      - 전체 데이터에 대한 평균 NDCG를 담은 dict
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    # 만약 datasets가 dict (컬럼 단위의 데이터)라면 리스트-of-dict 형태로 변환
    if isinstance(datasets, dict):
        logging.info("입력 데이터셋이 dict 형태입니다. 리스트 형태로 변환합니다.")
        keys = list(datasets.keys())
        n_samples = len(datasets[keys[0]])
        converted = []
        for i in range(n_samples):
            sample = { key: datasets[key][i] for key in keys }
            converted.append(sample)
        datasets = converted
        logging.info("총 %d개의 샘플로 변환되었습니다.", len(datasets))
    
    # 모델을 디바이스에 올림
    model.to(device)
    model.eval()
    
    # GPU 멀티 GPU 설정 (DataParallel 활용)
    if device == 'cuda' and torch.cuda.device_count() > 1 and gpu_num > 1:
        available_gpus = torch.cuda.device_count()
        use_gpus = min(gpu_num, available_gpus)
        if use_gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(use_gpus)))
            logging.info("Using %d GPUs in DataParallel.", use_gpus)
    
    ndcg_scores = []
    # DataLoader를 이용하여 데이터를 배치 처리합니다.
    datasets = datasets.shuffle(seed=42)
    datasets = datasets.select(range(10000))
    datasets = datasets.map(extract_text, fn_kwargs={"positive_field": positive_field, "negative_field": negative_field})
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Retriever (batched)", file=TqdmToLogger(logging.getLogger(__name__))):
            # 배치 내 각 sample의 type과 일부 내용을 로깅
            for idx, sample in enumerate(batch):
                logging.debug("Batch sample %d type: %s, content: %s", idx, type(sample), str(sample)[:100])
            
            # 각 샘플에서 query, positive, negative 필드를 추출 (예외 발생 시 로깅 후 중단)
            try:
                queries = batch[query_field]
                positives = batch[positive_field]
                negatives = batch[negative_field]
            except Exception as e:
                logging.error("에러 발생: 배치 내 필드 접근 중 오류 발생. Batch: %s, error: %s", str(batch), str(e))
                raise e
            
            # 리스트를 토큰화하여 배치 입력으로 변환 (padding, truncation 적용)
            query_inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)
            positive_inputs = tokenizer(positives, return_tensors="pt", padding=True, truncation=True).to(device)
            negative_inputs = tokenizer(negatives, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # 모델로부터 임베딩 추출 (일반적으로 CLS 토큰 사용)
            query_output = model(**query_inputs)
            pos_output = model(**positive_inputs)
            neg_output = model(**negative_inputs)
            query_emb = query_output.last_hidden_state[:, 0, :]
            pos_emb = pos_output.last_hidden_state[:, 0, :]
            neg_emb = neg_output.last_hidden_state[:, 0, :]
            
            # 배치별 코사인 유사도 계산
            pos_sim = torch.nn.functional.cosine_similarity(query_emb, pos_emb, dim=1)
            neg_sim = torch.nn.functional.cosine_similarity(query_emb, neg_emb, dim=1)
            
            # 각 샘플에 대해 NDCG 계산 (positive: relevance 1, negative: relevance 0)
            for ps, ns in zip(pos_sim.tolist(), neg_sim.tolist()):
                sim_scores = [ps, ns]
                true_labels = [1, 0]
                ndcg = compute_ndcg(sim_scores, true_labels)
                ndcg_scores.append(ndcg)
    
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    logging.info("전체 평균 NDCG: %f", avg_ndcg)
    return {'avg_ndcg': avg_ndcg}

# ---------------------------
def evaluate_sts(model, tokenizer, dataset, batch_size=32, s1='sentence1', s2='sentence2', label='labels', device='cuda'):
    """
    STS 태스크 평가 함수
    각 샘플은 두 문장(s1, s2)과 라벨(label)을 포함합니다.
    모델은 개별 문장에 대해 임베딩을 반환하며, 두 임베딩 간 코사인 유사도를 계산합니다.
    계산된 유사도와 ground truth 라벨 간의 Pearson, Spearman 상관계수 및 MSE를 반환합니다.
    """
    model.eval()
    model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    predictions = []
    ground_truths = []
    
    for batch in tqdm(dataloader, desc="Evaluating STS", file=TqdmToLogger(logging.getLogger(__name__))):
        # 문장과 라벨 추출
        sentences1 = batch[s1]
        sentences2 = batch[s2]
        labels = batch[label]
        
        # 라벨이 리스트이고 각 요소가 dict라면 원하는 키('label') 값 추출
        if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], dict):
            labels = [item["label"] for item in labels]
        
        # 라벨이 텐서가 아니라면 변환 (딕셔너리인 경우는 이미 위에서 처리되었음)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels['label'])
        labels = labels.to(device)
        
        # 개별 문장에 대해 토큰화 진행
        encodings1 = tokenizer(sentences1, padding=True, truncation=True, return_tensors='pt')
        encodings2 = tokenizer(sentences2, padding=True, truncation=True, return_tensors='pt')
        encodings1 = {key: value.to(device) for key, value in encodings1.items()}
        encodings2 = {key: value.to(device) for key, value in encodings2.items()}
        
        with torch.no_grad():
            outputs1 = model(**encodings1)
            outputs2 = model(**encodings2)
        
        # 임베딩 추출: pooler_output가 있으면 사용, 없으면 CLS 토큰(첫 토큰) 사용
        if hasattr(outputs1, "pooler_output"):
            emb1 = outputs1.pooler_output
        else:
            emb1 = outputs1[0][:, 0]
            
        if hasattr(outputs2, "pooler_output"):
            emb2 = outputs2.pooler_output
        else:
            emb2 = outputs2[0][:, 0]
        
        # 두 임베딩 간 코사인 유사도 계산 (결과: (batch_size,) 텐서)
        cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
        
        # GPU 텐서를 CPU로 옮긴 후 리스트로 추가
        predictions.extend(cos_sim.cpu().numpy().tolist())
        ground_truths.extend(labels.cpu().numpy().tolist())
    
    # 리스트를 numpy 배열로 변환
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # 평가 지표 계산: Pearson, Spearman 상관계수 및 평균 제곱 오차 (MSE)
    pearson_corr, _ = pearsonr(predictions, ground_truths)
    spearman_corr, _ = spearmanr(predictions, ground_truths)
    mse = np.mean((predictions - ground_truths) ** 2)
    
    result = {'pearson_corr': pearson_corr, 'spearman_corr': spearman_corr, 'mse': mse} 
    return result

# ---------------------------
def evaluate_retriever_ndcg(model, tokenizer, all_datasets, s1, s2, s3, label, top_k=5, device='cuda', batch_size=32):
    """
    STS 태스크 평가 함수 (NDCG 적용 버전)
    
    각 샘플은 query_text, corpus_text, score를 포함하며,
    동일한 query_text에 대해 여러 개의 corpus_text 후보가 존재하는 경우,
    각 후보와의 cosine similarity를 계산하여 ranking한 후,
    ranking 내에서 relevance (score 1인 항목) 분포를 기준으로 NDCG를 평가합니다.
    
    Args:
        model: 임베딩을 반환하는 모델.
        tokenizer: 문장을 토큰화하는 토크나이저.
        all_datasets: 평가에 사용할 데이터셋들을 포함하는 딕셔너리.
        device (str): 모델과 데이터를 올릴 디바이스 (기본값 'cuda').
        batch_size (int): 배치 크기 (기본값 32).
    
    Returns:
        avg_ndcg (float): 전체 query에 대한 평균 NDCG.
    """
    logging.info("evaluate_retriever_ndcg 함수 시작")
    # 각 데이터셋에서 데이터프레임 추출 (여기서는 test 데이터셋이 있을 경우 사용)
    if "test" in all_datasets[s1]:
        queries_df = all_datasets[s1]["test"].to_pandas()
        logging.info("queries_df: test split 사용")
    else:
        queries_df = all_datasets[s1].to_pandas()
        logging.info("queries_df: 전체 데이터 사용")
    
    if "test" in all_datasets[s2]:
        corpus_df = all_datasets[s2]["test"].to_pandas()
        logging.info("corpus_df: test split 사용")
    else:
        corpus_df = all_datasets[s2].to_pandas()
        logging.info("corpus_df: 전체 데이터 사용")
    
    if "test" in all_datasets[s3]:
        default_df = all_datasets[s3]["test"].to_pandas()
        logging.info("default_df: test split 사용")
    else:
        default_df = all_datasets[s3].to_pandas()
        logging.info("default_df: 전체 데이터 사용")
    
    # 컬럼명 변경
    queries_df = queries_df.rename(columns={'text': 'query_text'})
    corpus_df = corpus_df.rename(columns={'text': 'corpus_text'})
    logging.info("컬럼명 변경 완료: query_text와 corpus_text")
    
    # 데이터셋 병합
    merged_df = default_df.merge(queries_df, left_on='query-id', right_on='_id', how='left')
    merged_df = merged_df.merge(corpus_df, left_on='corpus-id', right_on='_id', how='left')
    logging.info("데이터셋 병합 완료, 총 %d개의 레코드", len(merged_df))
    
    # 필요한 컬럼만 선택
    merged_df = merged_df[['query_text', 'corpus_text', 'score']]
    merged_records = merged_df.to_dict(orient='records')
    
    # CustomDataset과 custom_collate_fn은 미리 정의되어 있다고 가정합니다.
    dataset = CustomDataset(merged_records)
    
    model.eval()
    model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    
    results = []
    
    for batch in tqdm(dataloader, desc="Evaluating STS with NDCG", file=TqdmToLogger(logging.getLogger(__name__))):
        # 배치에서 query와 corpus 텍스트, 라벨(score) 추출
        queries = batch['query_text']
        corpus_texts = batch['corpus_text']
        labels = batch[label]
        
        # 형식 변환
        queries = [q if isinstance(q, str) else str(q) for q in queries]
        corpus_texts = [ct if isinstance(ct, str) else str(ct) for ct in corpus_texts]
        labels = [l if isinstance(l, int) else int(l) for l in labels]
        if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], dict):
            labels = [item["label"] for item in labels]
        
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        labels = labels.to(device)
        
        # 토큰화
        encodings_query = tokenizer(queries, padding=True, truncation=True, return_tensors='pt')
        encodings_corpus = tokenizer(corpus_texts, padding=True, truncation=True, return_tensors='pt')
        encodings_query = {k: v.to(device) for k, v in encodings_query.items()}
        encodings_corpus = {k: v.to(device) for k, v in encodings_corpus.items()}
        
        with torch.no_grad():
            outputs_query = model(**encodings_query)
            outputs_corpus = model(**encodings_corpus)
        
        # 임베딩 추출
        if hasattr(outputs_query, "pooler_output"):
            emb_query = outputs_query.pooler_output
        else:
            emb_query = outputs_query[0][:, 0]
            
        if hasattr(outputs_corpus, "pooler_output"):
            emb_corpus = outputs_corpus.pooler_output
        else:
            emb_corpus = outputs_corpus[0][:, 0]
        
        # cosine similarity 계산
        cos_sim = torch.nn.functional.cosine_similarity(emb_query, emb_corpus, dim=1)
        
        batch_results = [{
            'query_text': q,
            'cosine_sim': sim.item(),
            'score': int(l.item())
        } for q, sim, l in zip(queries, cos_sim, labels)]
        results.extend(batch_results)
    
    # 그룹별 NDCG 계산
    ndcg_scores = []
    query_groups = {}
    for entry in results:
        query = entry['query_text']
        if query not in query_groups:
            query_groups[query] = []
        query_groups[query].append(entry)
    
    for query, items in query_groups.items():
        ranked_items = sorted(items, key=lambda x: x['cosine_sim'], reverse=True)[:top_k]
        predicted_relevances = [item['score'] for item in ranked_items]
        # compute_ndcg 함수를 사용 (compute_ndcg는 compute_dcg와 유사하게 동작)
        dcg = compute_ndcg(predicted_relevances)
        
        ideal_items = sorted(items, key=lambda x: x['score'], reverse=True)[:top_k]
        ideal_relevances = [item['score'] for item in ideal_items]
        idcg = compute_ndcg(ideal_relevances)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    logging.info("evaluate_retriever_ndcg 완료 - 평균 NDCG: %f", avg_ndcg)
    return {'avg_ndcg': avg_ndcg}