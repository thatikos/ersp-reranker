import json
from datasets import Dataset, load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer


#Load model and tokenizer
nqreranker = 'google-bert/bert-base-uncased'    
tokenizer = BertTokenizer.from_pretrained(nqreranker)
model = BertForSequenceClassification.from_pretrained(nqreranker, num_labels=2)

#format dataset
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

with open('nq_train_with_scores.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

def reformat(instance):
    question = instance["question"]
    documents = instance["ctxs"]
    
    reformatted_instances = []
    
    for doc in documents[:16]:
        text = f"{question} {tokenizer.sep_token} {doc['title']} {doc['text']}"
        label = doc.get("fid_score", None)
        if label is not None:  # Only include contexts with non-null labels
            reformatted_instances.append({"label": label, "text": text})
    
    return reformatted_instances

reformatted_data = []

for datapoint in data:
    reformatted_datapoint = reformat(datapoint)
    reformatted_data.extend(reformatted_datapoint)


with open('reformatted_nq_with_scores_16.jsonl', 'w') as file:
    json.dump(reformatted_data, file, indent=2)