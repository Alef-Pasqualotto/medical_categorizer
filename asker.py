# Arquivo para, após treinamento, realizar perguntas

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {device}")

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

output_dir = 'results/question_categorizer'
tokenizer_loaded = AutoTokenizer.from_pretrained(output_dir)
model_loaded = AutoModelForSequenceClassification.from_pretrained(output_dir)
model_loaded.to(device)
model_loaded.eval() 

def predict_specialty(question, tokenizer, model, especialidades_list):
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()

    # Mapear a previsão numérica de volta para o nome da especialidade
    if prediction < len(especialidades_list):
        return especialidades_list[prediction]
    else:
        return f"Especialidade numérica {prediction} fora do alcance."

# Exemplo de uso:

questions = ["Pergunta 1", "Pergunta 2"]

for q in questions:
    predicted_specialty = predict_specialty(q, tokenizer_loaded, model_loaded, especialidades)
    print(f"Pergunta: {q}")
    print(f"Especialidade prevista: {predicted_specialty}")