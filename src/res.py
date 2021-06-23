import glob,argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="", type=str)

args = parser.parse_args()
print('args==>', args)


for aug in  ['baseline', 'eda', 'bt', 'gpt2_3', 'cbert','cmodbert','cmodbertp']:
    files = glob.glob('./{}_*_bert_{}.log'.format(args.dsn, aug))

    accs = []
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                if 'Final' in line:
                    acc = float(line.split()[-1])
                    accs.append(acc)

    assert len(accs) ==  len(files)
    print(aug, 'iter:', len(accs), 'mean acc==>', np.array(accs).mean())          