import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# Merge all predictions into a single DataFrame
def merge_prediction_files(path):
    files = [f for f in os.listdir(path) if f.endswith('predictions.xlsx')]
    all_scores_pred = [] 
    for f_name in files:
        df = pd.read_excel(path+'/'+f_name)
        settings = f_name.split('_')
        df['model'] = len(df) * [settings[0]]
        df['Task'] =  len(df) * [settings[1]]
        df['Dataset'] =  len(df) * [settings[2]]
        all_scores_pred.extend(df.to_dict("records"))

    return pd.DataFrame(all_scores_pred)


# Calculate the specificity, precision, recall, and F1 score for the given model.
def calc_scores_binary_class(df_all):
    sizes = [1,2,4,8,16,32,64]

    f1_scores = []
    precision_scores = []
    recall_scores = []
    specificity_scores = []

    for t in ['f', 'q']:

        for d in df_all['Dataset'].unique():
            
            for s in sizes:
                
                column_pred = "label_" + str(s)
                if t == 'f':
                    column_real = "IsFunctional"
                elif t == 'q':
                    column_real = "IsQuality"
                df_per_size = df_all[(df_all['Task'] == t) & (df_all['Dataset'] == d)][['RequirementText', column_real, column_pred]]
                
                precision = precision_score(df_per_size[column_real], df_per_size[column_pred])
                recall = recall_score(df_per_size[column_real], df_per_size[column_pred]) 
                f1 = f1_score(df_per_size[column_real], df_per_size[column_pred])
                tn, fp, fn, tp = confusion_matrix(df_per_size[column_real], df_per_size[column_pred]).ravel()

                specificity_scores.append({'task': t, 'size': s, 'dataset': d,'specificity': tn / (tn+fp)})
                precision_scores.append({'task': t, 'size': s, 'dataset': d,'precision': precision})
                recall_scores.append({'task': t,'size': s, 'dataset': d, 'recall': recall}) 
                f1_scores.append({'task': t,'size': s, 'dataset': d,'f1': f1}) 

    return pd.DataFrame(precision_scores), pd.DataFrame(recall_scores), pd.DataFrame(f1_scores), pd.DataFrame(specificity_scores)

# Optional: plot the predictions as part of a box plot  
def plot_boxplot(df, setting):
    sns.set(style="whitegrid", context="notebook", font_scale=1.5)

    plt.figure(figsize=(10, 7))
    ax =sns.boxplot(x="size", y="score",  hue="task",data=df,  linewidth=2, palette="colorblind", width=0.6, showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black"})

    for spine in ax.spines.values():
        spine.set_linewidth(2.5) 

    plt.xlabel("Batch Size", fontsize=18, fontweight='bold')
    plt.ylabel("F1-score", fontsize=18, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    
    title_patch = mpatches.Patch(color='none', label='Requirement type')

    handles = [title_patch] + handles
    labels = ['Requirement type:', 'functional', 'quality']

    ax.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.42, 1.15),
        ncol= 4,
        frameon=True,      
        framealpha=0.1,     
        edgecolor='black', 
        facecolor='white', 
    )
    plt.savefig(setting+"_boxplot.pdf", format='pdf')


def main():

    path = r"final_predictions"

    gemma_folder = '/gemma12'
    ds14_folder = '/deepseek14'
    llama_folder = '/llama8'

    all_scores = {"f1":[], "precision": [], "recall": [], 'specificity': []}

    # For each model, merge the predictions and calculate the specificity, precision, recall, and F1 scores.
    for f in [gemma_folder, llama_folder, ds14_folder]:
        if os.path.exists(path+f):

            df_all = merge_prediction_files(path+f)
            
            df_all = df_all[df_all['Dataset'].isin(['l', 'p','rds', 'o'])]
            print(df_all['Dataset'].unique())
            df_precision, df_recall, df_f1, df_specificity = calc_scores_binary_class(df_all)
            
            df_all.loc[:, ~df_all.columns.str.contains('^Unnamed')].to_excel(path+'/'+f[1:]+"_all_pred.xlsx")

            df_specificity_flat = df_specificity.melt(id_vars=['size', 'dataset',  'task'], value_vars=['specificity'], var_name='scores',value_name='score')
            df_f1_flat = df_f1.melt(id_vars=['size', 'dataset',  'task'], value_vars=['f1'], var_name='scores',value_name='score')
            df_precision_flat = df_precision.melt(id_vars=['size', 'dataset',  'task'], value_vars=['precision'], var_name='scores',value_name='score')
            df_recall_flat = df_recall.melt(id_vars=['size', 'dataset',  'task'], value_vars=['recall'], var_name='scores',value_name='score')
            
            #plot_boxplot(df_f1_flat,path+'/' +f[1:])
                
            df_f1_flat['model'] = [f[1:]]*len(df_f1_flat)
            df_recall_flat['model'] = [f[1:]]*len(df_recall_flat)
            df_precision_flat['model'] = [f[1:]]*len(df_precision_flat)
            df_specificity_flat['model'] = [f[1:]]*len(df_specificity_flat)

            all_scores['f1'].extend(df_f1_flat.to_dict(orient= 'records'))
            all_scores['recall'].extend(df_recall_flat.to_dict(orient= 'records'))
            all_scores['precision'].extend(df_precision_flat.to_dict(orient= 'records'))
            all_scores['specificity'].extend(df_specificity_flat.to_dict(orient= 'records'))  
            
            df_f1_flat.pivot(index='size', columns=['dataset', 'task'], values=['score']).to_excel(path+f+'f1_table.xlsx')
    
    pd.DataFrame(all_scores['specificity']).to_excel(path+'/'+'all_specificity_score.xlsx')
    pd.DataFrame(all_scores['f1']).to_excel(path+'/'+'all_f1_score.xlsx')
    pd.DataFrame(all_scores['recall']).to_excel(path+'/'+'all_recall_score.xlsx')    
    pd.DataFrame(all_scores['precision']).to_excel(path+'/'+'all_precision_score.xlsx')        

if __name__ == '__main__':
    main()