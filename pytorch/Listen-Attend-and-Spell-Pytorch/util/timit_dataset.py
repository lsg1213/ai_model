from six.moves import cPickle
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

def load_dataset(data_path,**kwargs):
    with open(data_path, 'rb') as cPickle_file:
        [X_train, y_train, X_val, y_val, X_test, y_test] = cPickle.load(cPickle_file)
    for data in [X_train, y_train, X_val, y_val, X_test, y_test]:
        assert len(data) > 0
    return X_train, y_train, X_val, y_val, X_test, y_test

# Input x: list of np array with shape (timestep,feature)
# Return new_x : a np array of shape (len(x), padded_timestep, feature)
def ZeroPadding(x,pad_len):
    features = x[0].shape[-1]
    new_x = np.zeros((len(x),pad_len,features))
    for idx,ins in enumerate(x):
        new_x[idx,:len(ins),:] = ins
    return new_x

# A transfer function for LAS label
# We need to collapse repeated label and make them onehot encoded
# each sequence should end with an <eos> (index = 1)
# Input y: list of np array with shape ()
# Output tuple: (indices, values, shape)
def OneHotEncode(Y,max_len,max_idx=61):
    new_y = np.zeros((len(Y),max_len,max_idx+2))
    for idx,label_seq in enumerate(Y):
        last_value = -1
        cnt = 0
        for label in label_seq:
            if last_value != label:
                new_y[idx,cnt,label+2] = 1.0
                cnt += 1
                last_value = label
        new_y[idx,cnt,1] = 1.0 # <eos>
    return new_y

class TimitDataset(Dataset):
    def __init__(self, X, Y, max_timestep, max_label_len,bucketing):
        if not bucketing:
            self.X = ZeroPadding(X,max_timestep)
            self.Y = OneHotEncode(Y,max_label_len)
        else:
            batch_size = max_timestep
            bucket_x = []
            bucket_y = []
            sortd_len = [len(t) for t in X]
            sorted_x = [X[idx] for idx in reversed(np.argsort(sortd_len))]
            sorted_y = [Y[idx] for idx in reversed(np.argsort(sortd_len))]
            for b in range(int(np.ceil(len(sorted_x)/batch_size))):
                left = b*batch_size
                right = (b+1)*batch_size if (b+1)*batch_size<len(sorted_x) else len(sorted_x)
                pad_len = len(sorted_x[left]) if (len(sorted_x[left]) % 8) == 0 else\
                          len(sorted_x[left])+(8-len(sorted_x[left])%8)
                bucket_x.append(ZeroPadding(sorted_x[left:right], pad_len))
                bucket_y.append(OneHotEncode(sorted_y[left:right], max_label_len))
            self.X = bucket_x
            self.Y = bucket_y
    def __getitem__(self, index):
        return self.X[index],self.Y[index]
    def __len__(self):
        return len(self.X)


def create_dataloader(X, Y, max_timestep, max_label_len, batch_size, shuffle, bucketing, **kwargs):
    if not bucketing:
        return DataLoader(TimitDataset(X,Y,max_timestep,max_label_len,bucketing), 
                          batch_size=batch_size,shuffle=shuffle)
    else:
        return DataLoader(TimitDataset(X,Y,batch_size,max_label_len,bucketing), 
                          batch_size=1,shuffle=shuffle)
