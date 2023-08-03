import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

dna = ["ACGTAGCATCGGATCTATTTAGC", "ACGTAGCATCGGATCTATCTATCGACACCTATCATCTCGTTAGC", "ACGTATTATCGATCTACGAGCATCTCGTTAGC"]
inputs = tokenizer(dna, return_tensors='pt', padding=True)["input_ids"]
hidden_states = model(inputs)[0]  # [1, sequence_length, 768]

# torch.Size([3, 13, 768])
print(hidden_states.shape)
