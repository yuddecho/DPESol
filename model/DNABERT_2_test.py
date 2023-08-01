import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

dna = ["ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC", "ACGTAGCATCGGATCTATCTATCGACACTTGGCATCTCGTTAGC", "ACGTAGCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"]
inputs = tokenizer(dna, return_tensors='pt', padding=True)["input_ids"]
hidden_states = model(inputs)[0]  # [1, sequence_length, 768]
print(hidden_states.shape)

# embedding with mean pooling
embedding_mean = torch.mean(hidden_states, dim=1)
print(embedding_mean.shape)  # expect to be 768

# embedding with max pooling
embedding_max = torch.max(hidden_states[0], dim=0)[0]
print(embedding_max.shape)  # expect to be 768
