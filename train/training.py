from data_utils import *
from weakly_supervised_classifier import *
from filtering_chain import *

import os
import yaml

def main():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
        
    device = config['Full_pipeline']['Training_scheme']['device']
    config['Filtering_Chain']['Output_dir'] = config['Full_pipeline']['Training_scheme']['Output_dir']
    config['Weakly_Supervised']['Training_scheme']['Output_dir'] = config['Full_pipeline']['Training_scheme']['Output_dir']
    
    
    # Loading the datasets
    training_dataset_aes = making_training_sets()
    testing_dataset_aes = making_testing_sets()
    
    training_dataset_aes[5] = testing_dataset_aes
    
    training_dataset_wscs = {}
    training_dataset_wscs[0] = np.concatenate((training_dataset_aes[0], training_dataset_aes[1]), axis = 0)
    training_dataset_wscs[1] = training_dataset_aes[2]
    training_dataset_wscs[2] = training_dataset_aes[3]
    training_dataset_wscs[3] = training_dataset_aes[4]
    
    training_dataset_wscs[4] = Series_training(training_dataset_aes, config_dict=config['Filtering_Chain'])
    
    trainSeriesSupC_struct(training_dataset_wscs, config_dict=config['Weakly_Supervised'])
    
    return 0


if __name__ == "__main__":
    main()