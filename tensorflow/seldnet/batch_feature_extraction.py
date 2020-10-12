# Extracts the features, labels, and normalizes the training and test split features. Make sure you update the location
# of the downloaded datasets before in the cls_feature_class.py

import cls_feature_class

dataset_name = 'ansim'  # Datasets: ansim, resim, cansim, cresim, real, mansim and mreal

# Extracts feature and labels for all overlap and splits
for nffto in [512]: # For now use 512 point FFT. Once you get the code running, you can play around with this.
    for train in [True, False]:
        feat_cls = cls_feature_class.FeatureClass(nfft=nffto, train=train)

        # Extract features and normalize them
        feat_cls.extract_all_feature()
        feat_cls.preprocess_features()

        # # Extract labels in regression mode
        # feat_cls.extract_all_labels('regr', 0)
