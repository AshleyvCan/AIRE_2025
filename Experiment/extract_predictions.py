import pandas as pd
import ast
import os
import re
import sys
from sklearn.metrics import precision_score, recall_score, f1_score

def get_pred(chat_output, size, output_num = 2):
    matches = re.findall(r'(\d+)\s*\.?\s*=\s*(\d+)\s*', chat_output[output_num]['content'])

    return [{'id': int(id_), 'label_'+ str(size): int(label)} for id_, label in matches], [id_ for id_, _ in matches]

def extract_results(files, output_folder_size, size):
    data = []
    all_output = []
    for f_name in files:
        
        with open(output_folder_size+f_name, "r", encoding='utf-8') as f:
            chat_output = [ast.literal_eval(item) for item in f.readlines()]
            result, matches = get_pred(chat_output, size, 4)
            
            if len(result) != size or len(set(matches)) != size:
                result, matches = get_pred(chat_output, size)
                
                if len(result) != size:
                    print("INCORRECT TEMPLATE with" + f_name)
                    result = result[:size]
            all_output.extend(result)

        df_pred = pd.DataFrame(all_output)
        df_pred.set_index('id', inplace=True)
    
    return df_pred

def main(task, dataset, model):

    dataset_path = "datasets/"
    if dataset == 'l':
        df_all = pd.read_csv(dataset_path+'leeds_sample.csv')
    elif dataset == 'rds':
        df_all = pd.read_csv(dataset_path+'rds_sample.csv')
    elif dataset == 'o':
        df_all = pd.read_csv(dataset_path+'oappt_sample.csv')
    elif dataset == 'p':
        df_all = pd.read_csv(dataset_path+'promise_sample.csv')


    if task == 'f':
        column_label = 'IsFunctional'
    elif task == 'q':
        column_label = 'IsQuality'

    if model == 'ds7':
        pred_folder = 'deepseek7/'
        name_model = "deeps"
    elif model == 'ds14':
        pred_folder = 'deepseek14/'
        name_model = "deeps"
    elif model == 'llama':
        pred_folder = 'llama8/'
        name_model = "meta-"
    elif model == 'gemma':
        pred_folder = 'gemma12/'
        name_model = "googl"
    
    input_folder = 'llm_output/'
    output_folder = 'final_predictions/'
    
    if not os.path.exists(output_folder+ pred_folder):
        os.makedirs(output_folder+ pred_folder)

    sizes = [1,2,4,8,16,32,64]

    all_precision = []
    all_recall = []
    all_f1_score = []

    setting_name = '_'.join([name_model, task, dataset])
    for s in sizes:
        
        pred_folder_size =  '_'.join([str(s), setting_name])
        pred_path = input_folder + pred_folder +pred_folder_size+'/'
        files = [f for f in os.listdir(pred_path) if os.path.isfile(os.path.join(pred_path, f)) and f.startswith('chat_')]
        
        df_pred = extract_results(files, pred_path, s).sort_index()
        
        df_all = pd.merge(df_all, df_pred, left_index=True, right_index=True, how = 'left').fillna(-1)
        print(len(df_all['label_' + str(s)]))
        precision = precision_score(df_all[column_label], df_all['label_' + str(s)], average=None, labels=[0, 1])
        recall = recall_score(df_all[column_label], df_all['label_'+ str(s)], average=None, labels=[0, 1])
        f1= f1_score(df_all[column_label], df_all['label_'+ str(s)], average= None, labels=[0, 1] )

        all_precision.append({'size': s, 'precision_0': precision[0], 'precision_1': precision[1]})
        all_recall.append({'size': s, 'recall_0': recall[0], 'recall_1': recall[1]})
        all_f1_score.append({'size': s, 'f1_0': f1[0], 'f1_1': f1[1]})

    pd.DataFrame(all_precision).to_excel(output_folder+pred_folder+setting_name+"_"+"precision.xlsx")
    pd.DataFrame(all_recall).to_excel(output_folder+pred_folder+setting_name+"_"+"recall.xlsx")
    pd.DataFrame(all_f1_score).to_excel(output_folder+pred_folder+setting_name+"_"+"f1_score.xlsx")
    
    df_all.to_excel(output_folder+pred_folder+setting_name+"_"+"predictions.xlsx")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Arguments are missing. Follow the template: python extract_predictions.py <task letter: f or q> <dataset letter: l, p, o or r> <model letter: ds7, ds14, llama or gemma>")
    else:
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        arg3 = sys.argv[3]
        print("Task: ", arg1)
        print("Dataset: ", arg2)
        print("Model: ", arg3)
    main(sys.argv[1], sys.argv[2],  sys.argv[3])        

