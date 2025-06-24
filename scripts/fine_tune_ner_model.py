# Save this as fine_tune_ner_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, Features, Sequence, Value
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import precision_recall_fscore_support

# Load your dataset (assumed preprocessed and tokenized in previous steps)
# For example:
dataset = load_your_dataset()  # Implement your dataset loading or assign from previous code

# Define label mappings
unique_labels = list(set([label for labels in dataset['labels'] for label in labels]))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Map labels in dataset to IDs
dataset = dataset.map(lambda x: {'labels': [[label2id[l] for l in lbls] for lbls in x['labels']]})

# Initialize tokenizer
model_name = 'xlm-roberta-base'  # Or other models
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize and align labels function (as in your notebook)
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'], truncation=True, is_split_into_words=True, padding='max_length', max_length=128
    )
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Split dataset into train/test
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)

# Define models
model_xlm = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(unique_labels))
# For other models, replace accordingly

# Set training args
training_args = TrainingArguments(
    output_dir='./model_output',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss'
)

# Define compute_metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label for label in label_row if label != -100] for label_row in labels]
    pred_labels = [[pred for pred, true in zip(pred_row, true_row) if true != -100]
                   for pred_row, true_row in zip(predictions, labels)]
    true_flat = [item for sublist in true_labels for item in sublist]
    pred_flat = [item for sublist in pred_labels for item in sublist]

    precision, recall, f1, _ = precision_recall_fscore_support(true_flat, pred_flat, average='weighted', zero_division=1)
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Initialize trainer
trainer = Trainer(
    model=model_xlm,
    args=training_args,
    train_dataset=train_test_split['train'],
    eval_dataset=train_test_split['test'],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model
trainer.save_model('./best_ner_model')

# Evaluate and visualize (your existing code for plots)
eval_results = trainer.evaluate()
print(eval_results)