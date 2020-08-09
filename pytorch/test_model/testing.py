import os


datalength = [40, 80]
before = [0, 40, 80, 120]
opt = ['adam','sgd']
mode = ['ts_S', 'sj_S', 'no_S']
for m in mode:
    for d in datalength:
        for b in before:
            for o in opt:
                lr = [0.01, 0.001] if o == 'adam' else [0.1, 0.01]
                for l in lr:
                    os.system(f'python train.py --b {b} --opt {o} --len {d} --lr {l} --mode {m}')