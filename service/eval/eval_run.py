import mteb
import os

from eval.evaluater import evaluation_mteb

MODEL_LIST = [
    "x2bee/ModernBERT-ecs-GIST",
    "BM-K/KoSimCSE-roberta-multitask",
    "dragonkue/BGE-m3-ko",
    "nlpai-lab/KURE-v1",
    "nlpai-lab/KoE5",
    "jinaai/jina-embeddings-v3",
    "klue/roberta-base",
    "klue/bert-base",
]

OUTPUT_DIR = "./result"

TASK_LIST = [
    "KLUE-TC",
    "MIRACLReranking",
    "SIB200ClusteringS2S",
    "KLUE-STS",
    "KorSTS",
    "KLUE-NLI",
    "PawsXPairClassification",
    "PubChemWikiPairClassification",
    "AutoRAGRetrieval",
    "Ko-StrategyQA",
    "BelebeleRetrieval",
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MrTidyRetrieval",
    "MultiLongDocRetrieval",
    "XPQARetrieval",
]

RERANKING_TASK_LIST = [
    "MIRACLReranking",
]

RETRIEVAL_TASK_LIST = [
    "AutoRAGRetrieval",
    "Ko-StrategyQA",
    "BelebeleRetrieval",
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MrTidyRetrieval",
    "MultiLongDocRetrieval",
    "XPQARetrieval",
]

CLS_TASK_LIST = [
    "KLUE-TC",
    "PawsXPairClassification",
    "PubChemWikiPairClassification",
]

STS_TASK_LIST = [
    "KLUE-STS",
    "KorSTS",
]

def mteb_eval_run(model_list, output_dir, num_batch:int = 16, task_type:str = None, dataset_name:str = None):
    if task_type in ['sts', 'sentence_similarity', 'sentence-similarity', 'semantic_textual_similarity', 'semantic-textual-similarity']:
        task = STS_TASK_LIST
    elif task_type in ['retrieval', 'rt', 'Retrieval']:
        task = RETRIEVAL_TASK_LIST
    elif task_type in ['cls', 'classification', 'class', 'classify']:
        task = CLS_TASK_LIST
    elif task_type in ['reranking', 'reranker', 'rerank']:
        task = RERANKING_TASK_LIST
    elif task_type in ['all', 'whole', 'kor_all', 'kor']:
        task = TASK_LIST
    else:
        raise ValueError(f"[ERROR] Unexpected Task Type: {task_type}")
    
    if type(model_list) == str:
        _ = []
        try:
            _.append(model_list)
            model_list = _
        except:
            try:
                model_list = model_list.split(",")
            except:
                raise ValueError(f"[ERROR] model_list should be 'list[str]'. Please Check: {model_list}")
    
    else:
        print(f"[INFO] MTEB BENCHMARK START. TASK LIST: {task}")
        evaluation_mteb(model_list=model_list, output_dir=output_dir, batch_size=num_batch, task_list=task)
        

if __name__ == '__main__':
    tasks = mteb.get_benchmark("MTEB(kor, v1)")
