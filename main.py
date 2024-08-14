print("Starting main.py")
from src.data import GutenbergData
print("Imports done - Running main function")

def main():
    gutenberg_data = GutenbergData(batch_size=16, max_length=512)
    tensor_dataloader = gutenberg_data.get_dataloader(split='train')
    print("Batch input_ids shape:", tensor_dataloader.dataset.tensors['input_ids'].shape)




if __name__ == '__main__':
    main()
