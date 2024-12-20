import os
from huggingface_hub import login
from datasets import load_dataset
import tiktoken

# This was for getting the data read in Hugging face datasets

# hf_token = os.getenv("HF_TOKEN")
# if hf_token:
#     login(hf_token)
# else:
#     raise ValueError("Hugging Face token not set.")



# dataset = load_dataset("SouthernCrossAI/Project_Gutenberg_Australia")

# tokenizer = tiktoken.get_encoding("cl100k_base")

# # Define a tokenization function
# def tokenize_function(example):
#     tokens = tokenizer.encode(example["Paragraph Text"])  # Tokenize without truncation
#     example["cl100k_base"] = tokens             # Add tokenized data as a new column
#     return example

# # Apply tokenization to the dataset
# tokenized_dataset = dataset.map(tokenize_function)

# # Verify the new column
# print(tokenized_dataset["test"][0])  # Inspect the first row of the dataset


# # Push updated dataset to Hugging Face
# tokenized_dataset.push_to_hub("SouthernCrossAI/Project_Gutenberg_Australia")

# if __name__ == "__main__":
#     print("--------Testing----------")

#     # Initialize the tokenizer
#     tokenizer = tiktoken.get_encoding("cl100k_base")

#     # Example token IDs
#     tokens = [ 46639, 526, 433, 682, 1359, 1099, 26902, 23750, 3258, 11, 330, 2465, 15369, 5084, 311, 617, 83410, 757, 13, 1115, 4131, 315, 3515, 264, 3361, 49837, 369, 264, 5333, 13, 358, 1176, 65047, 28883, 264, 11102, 13272, 11, 1243, 74537, 11, 323, 1457, 31685, 2011, 6445, 264, 30077, 26, 323, 3686, 17354, 358, 2733, 439, 3582, 358, 1436, 6140, 264, 2763, 315, 3958, 15369, 369, 279, 25491, 315, 423, 34364, 14637, 5721, 13, 358, 649, 956, 1781, 1268, 433, 374, 279, 3828, 61191, 757, 779, 13, 358, 5895, 994, 358, 4985, 1518, 1077, 1578, 30, 2030, 1070, 596, 279, 3070, 11, 323, 279, 7126, 13, 3092, 39526, 0, 433, 1053, 387, 279, 38736, 596, 1866, 15369, 311, 617, 311, 1650, 26757, 14637, 5721, 364, 23881, 3502, 31412, 3238, 358, 8434, 956, 656, 433, 369, 279, 72021, 6548, 11, 477, 10437, 478, 23726, 11, 477, 294, 1673, 13744, 4579, 11, 477, 8369, 478, 4851, 430, 3596, 574, 13, 2360, 11, 28146, 11, 1070, 596, 912, 32961, 26, 279, 17983, 6420, 315, 279, 7126, 1053, 25760, 279, 5030, 13, 358, 1436, 539, 1781, 315, 433, 13, 358, 2643, 422, 568, 1051, 5710, 1210 ]


#     # Decode token IDs into text
#     decoded_text = tokenizer.decode(tokens)

#     print("Decoded Text:", decoded_text)
    
#     print("--------Testing Compleat----------")
