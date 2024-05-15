import json
from datasets import Dataset, load_dataset
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AutoTokenizer
import torch
from torch import nn
from sklearn.model_selection import train_test_split

#Load model and tokenizer
nqreranker = 'google-bert/bert-base-uncased'    
tokenizer = BertTokenizer.from_pretrained(nqreranker)
model = BertForSequenceClassification.from_pretrained(nqreranker, num_labels=2)


#load data
from datasets import Dataset
data_list = []
with open("/project/pi_hzamani_umass_edu/asalemi/ERSP/reformatted_nq_with_scores_16.jsonl", "r") as file:
    data_list = (json.load(file))

data = Dataset.from_list(data_list)


#preprocess
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_data = data.map(preprocess_function, batched=True)


from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#train
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)


training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
    push_to_hub=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()


#Inference

infData = Dataset.from_list('/project/pi_hzamani_umass_edu/asalemi/ERSP/nq/nq_dev.json')
data_args = {
    'ctx_size': 100,  
    'output_address': 'output.json',  
    'output_format': 'json'  
}

torch.softmax(torch.tensor(trainer.predict(infData).predictions), dim = -1)[:,1].view(2837, data_args.ctx_size).tolist()
scores_pred = torch.softmax(torch.tensor(trainer.predict(infData).predictions), dim = -1)[:,1].view(2837, data_args.ctx_size).tolist()
    
new_dataset = []
for data, scores in zip(infData, scores_pred):
    for ctx, score in zip(data['ctxs'], scores):
        ctx['rerank_score'] = score
    data['ctxs'].sort(key=lambda x: x['rerank_score'], reverse=True)
    new_dataset.append(data)
    
with open(data_args.rerankeroutput.jsonl, 'w') as file:
    if data_args.output_format == "json":
        json.dump(new_dataset, file, indent = 4)
    elif data_args.output_format == "jsonl":
        for data in new_dataset:
            json.dump(data, file)
            file.write("\n")

