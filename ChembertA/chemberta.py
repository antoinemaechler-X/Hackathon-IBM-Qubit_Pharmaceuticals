from transformers import pipeline
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score


pipe = pipeline("fill-mask", model="DeepChem/ChemBERTa-77M-MLM")

data = pd.read_csv("data/train.csv")
smiles = data["smiles"]
y = data["class"]

# Example of loading and preparing the dataset
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
train_smiles, test_smiles, train_labels, test_labels = train_test_split(
    smiles, y, test_size=0.2, random_state=42
)

# Convert to a dictionary format for Hugging Face's Trainer
train_data = {"smiles": train_smiles.tolist(), "labels": train_labels.tolist()}
test_data = {"smiles": test_smiles.tolist(), "labels": test_labels.tolist()}


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        
    }

from torch.utils.data import Dataset
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer):
        self.labels = labels.reset_index(drop=True)

        self.tokenizer = tokenizer
        self.smiles_list = smiles_list.reset_index(drop=True)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(smiles, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Remove batch dimension
        inputs["labels"] = torch.tensor(label, dtype=torch.long)
        return inputs

train_dataset = SMILESDataset(train_smiles, train_labels, tokenizer)
test_dataset = SMILESDataset(test_smiles, test_labels, tokenizer)


from transformers import RobertaForSequenceClassification

# Load pre-trained ChemBERTa and configure it for classification
model = RobertaForSequenceClassification.from_pretrained(
    "DeepChem/ChemBERTa-77M-MLM",
    num_labels=2  # Assuming binary classification
)


from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",           # Directory to save checkpoints
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    learning_rate=1e-3,              # Initial learning rate
    per_device_train_batch_size=128,  # Batch size for training
    per_device_eval_batch_size=64,   # Batch size for evaluation
    num_train_epochs=10,              # Number of epochs
    weight_decay=0.005,               # Weight decay
    logging_dir="./logs",            # Logging directory
    logging_steps=50,                # Log every 50 steps
)

# Set up the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()


# Evaluate and calculate metrics
from sklearn.metrics import classification_report

predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(-1)

print(classification_report(test_labels, predicted_labels))
