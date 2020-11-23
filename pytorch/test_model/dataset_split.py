import pickle, os, pdb, joblib
import numpy as np

def getdataset(datapath, labelpath, splitratio=0.95):
    if os.path.basename(datapath).split('.')[-1] == 'pickle':
        data = pickle.load(open(datapath, 'rb'))
        label = pickle.load(open(labelpath, 'rb'))
    

    testdata = [i[int(splitratio * len(i)):] for i in data]
    newdata = [i[:int(splitratio * len(i))] for i in data]
    testlabel = [i[int(splitratio * len(i)):] for i in label]
    newlabel = [i[:int(splitratio * len(i))] for i in label]
    pdb.set_trace()
    joblib.dump(newdata, open('stationary_accel_train.joblib', 'wb'))
    joblib.dump(testdata, open('stationary_accel_test.joblib', 'wb'))
    joblib.dump(newlabel, open('stationary_sound_train.joblib', 'wb'))
    joblib.dump(testlabel, open('stationary_sound_test.joblib', 'wb'))


if __name__ == "__main__":
    data = getdataset('/home/skuser/stationary_accel_data.pickle', '/home/skuser/stationary_sound_data.pickle')