import os 
import json

from service.eval.classifier.classificaiton_eval import compute_classfication
from service.eval.embedding.embedding_eval import evaluate_retriever_ndcg, evaluate_sts, evaluate_retriever_np_ndvg
from service.eval.reranker.reranker_eval import eval_reranker
from service.eval.LLM.llm_evaluator import evaluate_causal_lm
from service.eval.LLM.lm_harness_evaluator import run_lm_eval

from tqdm.auto import tqdm

from huggingface_hub import HfApi, hf_hub_download
from minio import Minio
from minio.error import S3Error
from datasets import load_dataset, load_from_disk
from sentence_transformers import CrossEncoder

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoModelForCausalLM,
)

import logging
logging.basicConfig(level=logging.INFO)

def minio_client():
    client = Minio(
        "polar-store-api.x2bee.com",
        access_key = "W2EmgRFVCGzlQ8u5wUUW",
        secret_key = "NEfmJTEwDWm5XSyM6imBsl1QjrnmaZSAB37bRnDk",
        secure = True,
    )
    return client 

client = minio_client()

def HfCheck(model , name):
    try:
        if model:
            api = HfApi()
            api.model_info(name)
            return True
        else:
            api = HfApi()
            api.dataset_info(name)
            return True
    except:
        return False

def dataloader(dataset_name, dataset_minio_enabled):
    bucket_name = "data"      # 버킷 이름
    folder_prefix = dataset_name + '/'

    path = "./eval/minio/"
    download_dir = os.path.join(path, "dataset")
    if not dataset_minio_enabled:
        logging.info(f"[Dataset] HF Hub에서 '{dataset_name}' 데이터셋 로딩을 시작합니다.")
        dataset = load_dataset(dataset_name)
        logging.info(f"[Dataset] HF Hub에서 '{dataset_name}' 데이터셋 로딩에 성공했습니다.")
    else:
        try:
            logging.info(f"[Dataset] Minio에서 다운로드를 시작합니다. (Bucket: {bucket_name}, Prefix: {folder_prefix})")
            objects = client.list_objects(bucket_name, prefix=folder_prefix, recursive=True)
            for obj in objects:
                local_file_path = os.path.join(download_dir, obj.object_name)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                client.fget_object(bucket_name, obj.object_name, local_file_path)
                logging.info(f"[Dataset] {obj.object_name} 파일을 {local_file_path}로 다운로드 완료")
            dataset = load_from_disk(os.path.join(download_dir, dataset_name))
            logging.info(f"[Dataset] 로컬 디스크에서 '{os.path.join(download_dir, dataset_name)}' 데이터셋 로딩에 성공했습니다.")
        except S3Error as err:
            logging.error(" [Dataset] 다운로드 중 오류 발생: %s", err)
            raise err
    return dataset

def moelloder(model_name, model_type, minio_enabled, num_labels = None):
    bucket_name = 'models'
    folder_prefix = model_name + '/'
    path = "./eval/minio/"
    download_dir = os.path.join(path, "model")

    if not minio_enabled:
        logging.info(f"[Model] HF Hub에서 '{model_name}' 모델 로딩을 시작합니다.")
        api = HfApi()
        api.model_info(model_name)
        logging.info(f"[Model] HF Hub에서 '{model_name}' 모델 정보를 성공적으로 불러왔습니다.")
    else:
        try:
            logging.info(f"[Model] Minio에서 모델 다운로드를 시작합니다. (Bucket: {bucket_name}, Prefix: {folder_prefix})")
            objects = client.list_objects(bucket_name, prefix=folder_prefix, recursive=True)
            for obj in objects:
                local_file_path = os.path.join(download_dir, obj.object_name)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                client.fget_object(bucket_name, obj.object_name, local_file_path)
                logging.info(f"[Model] {obj.object_name} 파일을 {local_file_path}로 다운로드 완료")
            # 모든 파일 다운로드 후, 로컬 모델 경로로 변경
            model_name = os.path.join(download_dir, model_name)
            logging.info(f"[Model] 모든 모델 파일이 로컬 경로 '{model_name}'에 다운로드되었습니다.")
        except S3Error as err:
            logging.error(" [Model] 다운로드 중 오류 발생: %s", err)
            raise err
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_type == 'Classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    elif model_type in ['Semantic_Textual_Similarity', 'Retrieval']:
        model = AutoModel.from_pretrained(model_name)
    elif model_type in ['Reranking']:
        model = CrossEncoder(model_name)
    elif model_type == 'CausalLM':
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_type == 'CausalLM_task':
        model = model_name
    logging.info(f"[Model] '{model_name}' 모델과 토크나이저 로딩 완료")
    return model, tokenizer 

def evaluator_LM_task(job_name, model_name, dataset_name, gpu_count, model_minio_enabled, base_model=None):
   bucket_name = 'models'
   folder_prefix = model_name + '/'
   path = "./eval/minio/"
   download_dir = os.path.join(path, "model")
   
   # 메인 모델 다운로드
   if model_minio_enabled:
       try:
           logging.info(f"[Model] Minio에서 모델 다운로드를 시작합니다. (Bucket: {bucket_name}, Prefix: {folder_prefix})")
           objects = client.list_objects(bucket_name, prefix=folder_prefix, recursive=True)
           for obj in objects:
               local_file_path = os.path.join(download_dir, obj.object_name)
               os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
               client.fget_object(bucket_name, obj.object_name, local_file_path)
               logging.info(f"[Model] {obj.object_name} 파일을 {local_file_path}로 다운로드 완료")
           # 모든 파일 다운로드 후, 로컬 모델 경로로 변경
           model_name = os.path.join(download_dir, model_name)
           logging.info(f"[Model] 모든 모델 파일이 로컬 경로 '{model_name}'에 다운로드되었습니다.")
       except S3Error as err:
           logging.error(" [Model] 다운로드 중 오류 발생: %s", err)
           raise err
   
   # Base 모델 처리
   if base_model is not None:
       logging.info(f"[Base Model] Base model이 지정되었습니다: {base_model}")
       
       # Hugging Face 모델인지 확인하는 함수
       def is_huggingface_model(model_name):
           try:
               from transformers import AutoConfig
               # Hugging Face Hub에서 모델 정보를 가져올 수 있으면 HF 모델
               AutoConfig.from_pretrained(model_name)
               return True
           except Exception as e:
               logging.info(f"[Base Model] {model_name}은 Hugging Face 모델이 아닙니다: {e}")
               return False
       
       # Base model이 Hugging Face 모델인지 확인
       if is_huggingface_model(base_model):
           logging.info(f"[Base Model] {base_model}은 Hugging Face 모델입니다. 직접 사용합니다.")
           # base_model을 그대로 사용 (Hugging Face에서 자동 다운로드)
       else:
           # Hugging Face 모델이 아니면 Minio에서 다운로드
           logging.info(f"[Base Model] {base_model}은 Hugging Face 모델이 아닙니다. Minio에서 다운로드를 시도합니다.")
           
           base_folder_prefix = base_model + '/'
           base_download_dir = os.path.join(path, "base_model")
           
           try:
               logging.info(f"[Base Model] Minio에서 Base 모델 다운로드를 시작합니다. (Bucket: {bucket_name}, Prefix: {base_folder_prefix})")
               
               # Base 모델 다운로드 디렉토리 생성
               os.makedirs(base_download_dir, exist_ok=True)
               
               # Minio에서 base model 객체 목록 가져오기
               base_objects = client.list_objects(bucket_name, prefix=base_folder_prefix, recursive=True)
               base_objects_list = list(base_objects)  # 리스트로 변환하여 확인
               
               if not base_objects_list:
                   logging.warning(f"[Base Model] Minio에서 {base_folder_prefix} 경로에 파일을 찾을 수 없습니다.")
                   logging.info(f"[Base Model] {base_model}을 Hugging Face 모델로 간주하고 직접 사용합니다.")
               else:
                   # Base 모델 파일들 다운로드
                   for obj in base_objects_list:
                       local_file_path = os.path.join(base_download_dir, obj.object_name)
                       os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                       client.fget_object(bucket_name, obj.object_name, local_file_path)
                       logging.info(f"[Base Model] {obj.object_name} 파일을 {local_file_path}로 다운로드 완료")
                   
                   # 로컬 base model 경로로 변경
                   base_model = os.path.join(base_download_dir, base_model)
                   logging.info(f"[Base Model] 모든 Base 모델 파일이 로컬 경로 '{base_model}'에 다운로드되었습니다.")
                   
           except S3Error as err:
               logging.error(f"[Base Model] Minio 다운로드 중 오류 발생: {err}")
               logging.info(f"[Base Model] {base_model}을 Hugging Face 모델로 간주하고 직접 사용합니다.")
               # 에러 발생 시 원본 base_model 이름을 그대로 사용 (HF 모델로 간주)
           except Exception as e:
               logging.error(f"[Base Model] 예상치 못한 오류 발생: {e}")
               logging.info(f"[Base Model] {base_model}을 Hugging Face 모델로 간주하고 직접 사용합니다.")
   
   # 평가 실행 (base_model이 있으면 함께 전달)
   if base_model is not None:
       logging.info(f"[Evaluation] Base model과 함께 평가를 실행합니다.")
       logging.info(f"[Evaluation] Main model: {model_name}, Base model: {base_model}")
       result = run_lm_eval(job_name, model_name, dataset_name, gpu_count)
       base_model_result = run_lm_eval(job_name, base_model, dataset_name, gpu_count, base_model = True)
       return result, base_model_result
   else:
       logging.info(f"[Evaluation] 단일 모델 평가를 실행합니다.")
       result = run_lm_eval(job_name, model_name, dataset_name, gpu_count)
   
   return result, None

def evaluator_LM(model_name, dataset_name, gpu_count, use_cot, model_minio_enabled, dataset_minio_enabled, max_new_tokens = 16, column1 = 'question', column2 = 'options',  column3 = 'cot_content', label = 'answer_index'):
    dataset = dataloader(dataset_name, dataset_minio_enabled)
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model, tokenizer = moelloder(model_name, model_type="CausalLM", minio_enabled=model_minio_enabled)
    
    result = evaluate_causal_lm(
        model = model, 
        tokenizer = tokenizer, 
        ds = dataset, 
        gpu_count = gpu_count, 
        max_new_tokens = max_new_tokens, 
        use_cot = use_cot, 
        question_field = column1, 
        options_field = column2, 
        cot_field = column3, 
        answer_idx_field = label)

    return result 

def evaluator_bert(task, model_name, dataset_name, column1, column2, column3, label, top_k, gpu_num, model_minio_enabled = True, dataset_minio_enabled = True):
    # 데이터셋 로딩
    dataset = dataloader(dataset_name, dataset_minio_enabled)
    if "test" in dataset:
        dataset = dataset['test']
    elif 'train' in dataset:
        dataset = dataset['train']        
    
    # 작업 타입에 따른 모델 로드
    if task == 'Classification':
        logging.info(f"[Evaluator] '{task}' 작업: 데이터셋 내 고유 라벨 개수 {len(set(dataset[label]))} 확인")
        unique_count = len(set(dataset[label]))
        model, tokenizer = moelloder(model_name, task, model_minio_enabled, unique_count)
    elif task in ['Semantic_Textual_Similarity', 'Retrieval']:
        model, tokenizer = moelloder(model_name, task, model_minio_enabled)
    elif task == 'Reranking':
        try:
            model, tokenizer = moelloder(model_name, task, model_minio_enabled)
        except Exception as e:
            # CrossEncoder 모델 로드 실패 시 fallback: Semantic_Textual_Similarity로 전환
            task = 'Semantic_Textual_Similarity'
            model, tokenizer = moelloder(model_name, task, model_minio_enabled)
    else:
        raise Exception("알 수 없는 작업 타입입니다.")
    
    # 평가 수행
    if task == 'Classification':
        result = compute_classfication(model, tokenizer, dataset, label, column1, top_k=top_k)
    elif task == 'Semantic_Textual_Similarity': 
        result = evaluate_sts(model, tokenizer, dataset, s1=column1, s2=column2, label=label)
    elif task == 'Retrieval':
        if label:
            result = evaluate_retriever_ndcg(model, tokenizer, dataset, s1=column1, s2=column2, s3=column3, label=label, top_k=top_k)
        else: 
            result = evaluate_retriever_np_ndvg(model, tokenizer, dataset, query_field=column1, positive_field=column2, negative_field=column3, gpu_num = gpu_num)
    elif task == 'Reranking':
        result = eval_reranker(model, dataset, at_k=top_k)
    
    return result 


