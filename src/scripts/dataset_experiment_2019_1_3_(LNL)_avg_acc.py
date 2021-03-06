import argparse
import sys
sys.path.append('C:/Users/YH/PycharmProjects/AE-signal-model')

from src.utils.helpers import *
parser = argparse.ArgumentParser(description='Input some parameters.')

parser.add_argument('--model', default=None, type=str, help='model name to test')
parser.add_argument('--rfname', default=1, type=str, help='Result File name')
parser.add_argument('--inlabel_unseen', default=None, type=str, nargs='+', help='input label')
parser.add_argument('--inlabel_seen', default=None, type=str, nargs='+', help='input label')
args = parser.parse_args()
MODEL_NAME_TO_TEST = args.model
RESULT_SAVE_FILENAME = args.rfname
INPUT_DATA_LABEL_UNSEEN = args.inlabel_unseen
INPUT_DATA_LABEL_SEEN = args.inlabel_seen

file_dir = direct_to_dir(where='result') + '{}_acc_buffer.csv'.format(MODEL_NAME_TO_TEST)
df = pd.read_csv(file_dir)
print(df)
# calc mean unseen score
unseen_mean_acc = np.sum([df['Unseen-Leak'].values, df['Unseen-NoLeak']], axis=0)
seen_mean_acc = np.sum([df['Seen-Leak'].values, df['Seen-NoLeak']], axis=0)

# append to the result files
with open(RESULT_SAVE_FILENAME, 'a') as f:
    f.write('\n\nOVERALL ACCURACY ---------------------- UNSEEN')
    for i, j in zip(INPUT_DATA_LABEL_UNSEEN, unseen_mean_acc):
        f.write('\n' + i + ' acc: {:.4f}'.format(j))

    f.write('\n\nOVERALL ACCURACY ---------------------- SEEN')
    for i, j in zip(INPUT_DATA_LABEL_SEEN, seen_mean_acc):
        f.write('\n' + i + ' acc: {:.4f}'.format(j))
