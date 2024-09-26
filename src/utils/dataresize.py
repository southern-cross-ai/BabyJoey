# ###############################################################################
#     # Uncomment to Reduce the datasets to 1/16 of their original size
#     def reduce_dataset(dataset, fraction):
#         dataset_size = len(dataset)
#         reduced_size = int(dataset_size * fraction)
#         indices = random.sample(range(dataset_size), reduced_size)
#         return Subset(dataset, indices)

#     training_dataset = reduce_dataset(training_dataset, 1/256)
#     validation_dataset = reduce_dataset(validation_dataset, 1/265)
# ###############################################################################