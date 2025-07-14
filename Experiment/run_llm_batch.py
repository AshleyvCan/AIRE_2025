import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
import pandas as pd
import os

def load_txt(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

def get_data(directory, prefix):
    return [f for f in os.listdir(directory) if f.startswith(prefix)]

def export_data(data,name ):
    with open(name, "w") as file:
        if isinstance(data, str):
            file.write(data)
        else:
            for item in data:
               file.write(f"{item}\n")

def main(stepsize, setting, dataset, m):
    torch.manual_seed(42)

    dataset_path = "datasets/"
    if dataset == 'l':
        df = pd.read_csv(dataset_path+'leeds_sample.csv')
    elif dataset == 'p':
        df = pd.read_csv(dataset_path+'promise_sample.csv')
    elif dataset == 'o':
        df = pd.read_csv(dataset_path+'oappt_sample.csv')
    elif dataset == 'rds':
        df = pd.read_csv(dataset_path+'rds_sample.csv')

    if m == 'gemma':
        model_id = "google/gemma-3-12b-it"
    elif m == 'llama':
        model_id = "meta-llama/Llama-3.1-8B-instruct"
    elif m == 'ds':
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

    print(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token= '<token>',)
    model = AutoModelForCausalLM.from_pretrained(model_id,token= '<token>'
                                                 )
    system_prompt = load_txt('prompt_system_' + setting + ".txt")
    user_prompt_1 = load_txt("prompt_user_1.txt")
    user_prompt_2 = load_txt("prompt_user_2.txt")

    output_dir = 'output/'

    n  = len(df)

    export_folder = output_dir + m +'/' + str(stepsize) + "_" + model_id[:5] +"_"+ setting + "_"+dataset+'/'
    
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    
    device = torch.cuda.current_device()
    model = model.to(device)
   
    all_reqs = [str(i) + ". " + row['RequirementText'] for i, row in df.iterrows()]
    for i in range(0,n,stepsize):
        if i+stepsize <= n:  
        
            req_data = '\n'.join(all_reqs[i:i+stepsize])

            message_history = [
                    {"role": "system", "content": system_prompt.format(m=i, n=i+(stepsize-1))},
                    {"role": "user", "content":user_prompt_1+"\n" + req_data}
                ]
        
            for j in range(2):

                formatted_chat = tokenizer.apply_chat_template(message_history, tokenize=False, add_generation_prompt=True)
                
                inputs = tokenizer(formatted_chat, return_tensors="pt") 
                inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
 
                outputs = model.generate(**inputs, max_length=10000, temperature = 0.01)

                decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
                message_history.extend([{"role": "assistant", "content": decoded_output}])         

                if j == 0: # after the first answer, reiterate the template to receive a final answer
                    message_history.append({'role': 'user', 'content': user_prompt_2.format(m=i, n=i+(stepsize-1))}) 

            export_data(message_history,export_folder+"chat_"+dataset+'_'+ str(stepsize)+'_'+str(i)+ '.txt')
            export_data(decoded_output,export_folder+"finaloutput"+dataset+'_'+ str(stepsize)+'_'+ str(i)  +".txt")

        
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Arguments are missing. Follow the template: python run_llm_batch.py <batchsize> <task letter: f or q> <dataset letter: l p or r> <model: llama gemma or ds>")
    else:
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        arg3 = sys.argv[3]
        arg4 = sys.argv[4]

        print("Batch size: ", arg1)
        print("Task: ", arg2)
        print("Dataset: ", arg3)
        print("Model: ", arg4)
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3],  sys.argv[4])