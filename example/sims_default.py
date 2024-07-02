import os
import pickle

from MSA_FET import FeatureExtractionTool
from MSA_FET import run_dataset

# initialize with default librosa config which only extracts audio features
fet = FeatureExtractionTool("sims_default")

# # alternatively initialize with a custom config file
# fet = FeatureExtractionTool("custom_config.json")

# extract features for single video
file = r'/home/drew/Documents/10.mp4'
feature1 = fet.run_single(file)
result_dir = 'result'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
result_path = os.path.join(result_dir, file.split('/')[-1].split('.')[0] + '.pkl')
pickle.dump(feature1, open(result_path, 'wb'))

# print(feature1)
# feature2 = fet.run_single("input2.mp4")

# # extract for dataset & save features to file
# run_dataset(
#     config = "aligned",
#     dataset_dir="~/MOSI", 
#     out_file="output/feature.pkl",
#     num_workers=4
# )