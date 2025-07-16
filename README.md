# Online Appendix of the paper "One Size Does Not Fit All: On the Role of Batch Size in Classifying Requirements with LLMs"

In this paper, we assess the impact of the selected batch size on the performance of an LLM.
The appendix contains two main folders: Tables and Experiments.

## Folder: Tables 
This folder contains the tables for comparing the models and qualitative analysis. In addition, this folder contains additional tables for a better understanding of the overall performance.

`qualitative_analysis_tables.xlsx` presents the tables of the qualitative analysis
`tables_performance_quality.xlsx` contains the data of table 6 to 8
`tables_performance_functional.xlsx` contains the same data of table 6 to 8, but for the functional classification task.


## Folder: Experiment
This folder contains materials to reproduce the experiment.

The prompts used in our experiments:
- `prompt_system_q.txt`: The system prompt for the classification of the quality requirements.
- `prompt_system_f.txt`: The system prompt for the classification of the functional requirements.
- `prompt_user_1.txt`: The first user prompt in the conversation.
- `prompt_user_2.txt`: After the first answer of the LLM, this prompt reiterates the template in order to receive a final answer.

The `run_llm_batch.py` is used to execute the formulated prompts, using a pre-defined batch size value. Use the following template to run the script:

```
python run_llm_batch.py <batchsize> <task letter: f or q> <dataset letter: l p, o or r> <model: llama gemma or ds>
```

The post-processing steps are conducted using `extract_predictions.py`. Run this script after producing the LLM outputs and follow the template: 

```
python extract_predictions.py <task letter: f or q> <dataset letter: l, p, o or r> <model letter: ds14, llama or gemma>
```

To calculate F1-score, recall, precision and specificity, the `get_performances.py` can be used. This script should also produce a combined Excel file in which the predictions for all datasets per model are merged(which are needed for the qualitative analysis and the ensemble model).

The file `qualitative_analysis.ipynb` produces the tables for the qualitative analysis.
The file `ensemble_analysis.ipynb` produces the results of the ensemble method and produces the tables for comparison.


### subfolder: Default batch size
Since the original prompt assumes that there are multiple requirements for the LLM, we have slightly modified the prompts for the use of a batch size 1.

- `prompt_system_q_single.txt`: The system prompt for the classification of the quality requirement (when using a batch size of 1)
- `prompt_system_f_single.txt`: The system prompt for the classification of the functional requirements (when using a batch size of 1)
- `prompt_user_1_single.txt`: The first user prompt in the conversation (when using a batch size of 1)
- `prompt_user_2_single.txt`: After the first answer of the LLM, this prompt reiterates the template in order to receive a final answer (when using a batch size of 1).


## Reference to data
Dell’Anna, D., Aydemir, F. B., & Dalpiaz, F. (2023). Evaluating classifiers in SE research: the ECSER pipeline and two replication studies. Empirical Software Engineering, 28(1), 3.

Dalpiaz, F., Dell'Anna, D., Aydemir, F. B., & Çevikol, S. (2019, September). Requirements classification with interpretable machine learning and dependency parsing. In 2019 IEEE 27th International Requirements Engineering Conference (RE) (pp. 142-152). IEEE.
