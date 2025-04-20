from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Cargar el modelo y tokenizer entrenados
model_path = "./models/deepfake-detector"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Usar GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Función de detección
def detect_text(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class == 0:
        return "🧠 Texto HUMANO"
    else:
        return "🤖 Texto GENERADO POR IA"

# Uso de ejemplo
if __name__ == "__main__":
    texto = input("Escribe el texto a analizar: ")
    resultado = detect_text(texto)
    print(f"\nResultado: {resultado}")
