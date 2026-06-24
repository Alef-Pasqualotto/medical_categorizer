# Arquivo de treinamento do modelo BERTimbau

import pandas as pd
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import KFold
import pandas as pd

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["pergunta"], max_length=256, truncation=True, padding='max_length')
    tokenized_inputs["labels"] = examples["especialidade_numerica"]
    return tokenized_inputs


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    f1_ma = f1_score(labels, preds, average='macro', zero_division=0)
    f1_mi = f1_score(labels, preds, average='micro', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1-macro': f1_ma,
        'f1-micro': f1_mi
    }


# Leitura dos dados
df_data = pd.read_csv('corpus_perguntas_respostas.csv', engine='python')
print(f"Total de perguntas: {len(df_data)}")
print(f"Número de perguntas únicas: {df_data['pergunta'].nunique()}")

# Remoção de duplicadas
duplicates = df_data[df_data.duplicated(subset=['pergunta'], keep=False)]

if not duplicates.empty:
    print("\nPerguntas duplicadas encontradas:")
    print(duplicates.sort_values(by='pergunta'))
else:
    print("\nNão foram encontradas perguntas duplicadas.")

print(f"Número de perguntas antes da remoção de duplicadas: {len(df_data)}")
df_data.drop_duplicates(subset=['pergunta'], keep='first', inplace=True)
print(f"Número de perguntas após remoção de duplicadas: {len(df_data)}")

# Numeração das especialidades
especialidades = []
def map_value(category):
    if category not in especialidades:
        especialidades.append(category)
        return len(especialidades) - 1
    else:
        return especialidades.index(category)

# Classificação das perguntas e filtragem por quantidade
limite = 50
counts = df_data['especialidade'].value_counts()
to_keep = counts[counts >= limite].index

df_data = df_data[df_data['especialidade'].isin(to_keep)]
df_data['especialidade_numerica'] = df_data["especialidade"].apply(map_value)

classes = df_data.especialidade.unique()
df_data.groupby(['especialidade', 'especialidade_numerica']).count()


# Configurações iniciais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {device}")


# Preparar o Tokenizer e Tokenizar os dados
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)


# Melhores hiperparâmetros encontrados
output_dir = 'results/question_categorizer'
epochs = 10
learning_rate = 3e-4
batch_size = 16
gradient_acc = 3
fp16 = 0

qtd_kfolds = 5
kf = KFold(n_splits=qtd_kfolds, shuffle=True, random_state=42)
indexes_kfold = [[0 for x in range(2)] for y in range(qtd_kfolds)]


for i, (train_index, test_index) in enumerate(kf.split(df_data)):
  indexes_kfold[i][0] = train_index
  indexes_kfold[i][1] = test_index

metricas = []
acc_cross = []
f1ma_cross = []
f1mi_cross = []
best_f1ma = -1

for i in range(0, qtd_kfolds):
    df_train = df_data[df_data.index.isin(indexes_kfold[i][0])]
    df_test = df_data[df_data.index.isin(indexes_kfold[i][1])]

    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)

    train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=['especialidade', 'resposta', 'avaliacao', 'curtidas', 'id', 'pergunta', 'especialidade_numerica'])
    test_dataset_tokenized = test_dataset.map(tokenize_function, batched=True, remove_columns=['especialidade', 'resposta', 'avaliacao', 'curtidas', 'id', 'pergunta', 'especialidade_numerica'])

    torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=len(classes))
    model.to(device)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy ='epoch',
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        logging_strategy='epoch',
        learning_rate=learning_rate,
        num_train_epochs = epochs,
        load_best_model_at_end = True,
        gradient_accumulation_steps=gradient_acc,
        fp16=(fp16==1)
    )
    
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenized,
    eval_dataset=test_dataset_tokenized,
    compute_metrics=compute_metrics,
    )

    trainer.train()

    p = trainer.predict(test_dataset_tokenized)
    metricas.append(p[2])

    predicoes = p[0].argmax(-1)
    true_values = list(test_dataset_tokenized['labels'])
    acc_cross.append(accuracy_score(true_values, predicoes))

    f1ma = f1_score(true_values, predicoes, average='macro', zero_division=0)
    f1ma_cross.append(f1ma)
    f1mi_cross.append(f1_score(true_values, predicoes, average='micro', zero_division=0))

    if f1ma > best_f1ma:
        best_f1ma = f1ma
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir, safe_serialization=False)

        # Save the test indices to a CSV
        pd.DataFrame({'test_index': test_index}).to_csv(f'{output_dir}/best_test_indices.csv', index=False)
        print(f"Fold {i+1}: novo melhor modelo salvo (f1ma={f1ma:.4f})")

import math
def show_metrics(name_metrica, metricas):
  mean = sum(metricas)/len(metricas)
  s = 0
  for data in range(0, len(metricas)):
    s = s + pow(metricas[data] - mean, 2)
  std_dev = math.sqrt(s/(len(metricas) - 1))
  print(f'{name_metrica}: {mean} ± {std_dev}')

show_metrics('Loss', [m['test_loss'] for m in metricas])
show_metrics('Acurácia', [acc for acc in acc_cross])
show_metrics('F1-Macro', [f1ma for f1ma in f1ma_cross])
show_metrics('F1-Micro', [f1mi for f1mi in f1mi_cross])

print(metricas)
