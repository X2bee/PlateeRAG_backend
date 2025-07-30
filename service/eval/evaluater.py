import mteb
import pandas as pd
import json

def evaluation_mteb(model_list:list[str], output_dir:str, batch_size:int=32, task_list:list[str]=None):
    preset = True
    
    if preset:
        tasks = mteb.get_benchmark("MTEB(kor, v1)")
    else:
        tasks = [(mteb.get_task(task, ['kor'])) for task in task_list]
        
    evaluation = mteb.MTEB(tasks=tasks)
    
    for n, model_name in enumerate(model_list):
        print("=" * 120)
        print("-"*30+f"{n}) {model_name} - Start"+"-"*30)
        
        try:
            model = mteb.get_model(model_name)
            evaluation.run(model, verbosity=3, output_folder=output_dir, encode_kwargs={"batch_size": batch_size})
            # evaluation.run(model, verbosity=3, output_folder=output_dir)
        except Exception as e:
            print(f'Error Occur: {model_name}')
            print(f'Error Occur: {e}')
            
    eval_data_list = [(item.metadata.name, item.metadata.type) for item in evaluation.tasks]
    columns = [a[0] for a in eval_data_list]
    result_df = pd.DataFrame(columns=columns)
    
    for n, model_name in enumerate(model_list):
        model_name_re = model_name.replace("/", "__")
        result = mteb.get_model_meta(model_name)
        revision_data = result.revision
        
        if revision_data == None:
            revision_data = "no_revision_available"
        
        output_path = f"{output_dir}/{model_name_re}/{revision_data}"
        model_metadata_path = f"{output_path}/model_meta.json"
        
        for task in eval_data_list:
            task_name, task_type = task
            json_item_path = f"{output_path}/{task_name}.json"
            with open(json_item_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if task_type == "Classification":
                value = data['scores']['validation'][0]['accuracy']
                
            if task_type == "Clustering":
                value = data['scores']['test'][0]['v_measure']
                
            if task_type == "PairClassification":
                try:
                    value = data['scores']['test'][0]['similarity_f1']
                except:
                    value = data['scores']['validation'][0]['similarity_f1']
                    
            if task_type == "Reranking":
                value = data['scores']['dev'][0]['NDCG@20(MIRACL)']
                
            if task_type == "Retrieval":
                try:
                    value = data['scores']['dev'][0]['ndcg_at_20']
                except:
                    try:
                        value = data['scores']['test'][0]['ndcg_at_20']
                    except:
                        value = data['scores']['validation'][0]['ndcg_at_20']
                    
            if task_type == "STS":
                try:
                    value = data['scores']['test'][0]['cosine_spearman']
                except:
                    value = data['scores']['validation'][0]['cosine_spearman']
                
            result_df.loc[model_name, task_name] = value
            
    result_df.to_csv(f"{output_dir}/total_result.csv", index=True, index_label="Model_Name")