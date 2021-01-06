# Extracts the features, labels, and normalizes the training and test split features. Make sure you update the location
# of the downloaded datasets before in the cls_feature_class.py

import cls_feature_class, pdb
from params import getparam
import sys
dataset_name = 'ansim'  # Datasets: ansim, resim, cansim, cresim, real, mansim and mreal
config = getparam(sys.argv[1:])
# Extracts feature and labels for all overlap and splits
for nffto in [512]: # For now use 512 point FFT. Once you get the code running, you can play around with this.
    for train in ['train', 'validation', 'test']:
        
        feat_cls = cls_feature_class.FeatureClass(nfft=nffto, datasettype=train, config=config)

        # Extract features and normalize them
        feat_cls.extract_all_feature()
        feat_cls.preprocess_features()

        # # Extract labels in regression mode
        feat_cls.extract_all_labels()
