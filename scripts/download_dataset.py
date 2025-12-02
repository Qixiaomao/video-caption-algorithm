from datasets import load_dataset

# using MSVD
ds = load_dataset("friedrichor/MSVD")

# print the dataset structure
print(ds['train'][0])