from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# 1. Cargar un dataset de prueba (puedes reemplazarlo más adelante con uno real)
dataset = load_dataset("rotten_tomatoes")  # Tiene texto y etiquetas 0/1
dataset = dataset.rename_column("label", "labels")  # Hugging Face espera que la columna se llame 'labels'

# 2. Cargar el tokenizer y el modelo preentrenado BERT
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Tokenizar los textos
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# 4. Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./models",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_strategy="epoch",  # Guardado por época
    logging_dir="./logs",
    logging_steps=10,
)

# 5. Crear el objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# 6. Entrenar el modelo
trainer.train()

# 7. Guardar el modelo entrenado
model.save_pretrained("./models/deepfake-detector")
tokenizer.save_pretrained("./models/deepfake-detector")

print("✅ Entrenamiento finalizado. Modelo guardado en 'models/deepfake-detector'.")
