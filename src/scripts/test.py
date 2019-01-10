import argparse
import time

time.sleep(2)

# argparse
parser = argparse.ArgumentParser(description='Input some parameters.')
parser.add_argument('--fts', metavar='FS', default=None, type=str, help='Filename to save')
parser.add_argument('--ftr', metavar='FR', default=None, type=str, help='Filename to process')
parser.add_argument('--cth', metavar='CH', default=None, type=int, nargs='+', help='Channel no to extract')
parser.add_argument('--svs', metavar='S', default=None, type=int, help='sample vector size')
parser.add_argument('--dsf', metavar='DF', default=1, type=int, help='Downsample factor')

args = parser.parse_args()

FILENAME_TO_SAVE = args.fts
FOLDER_TO_READ = args.ftr
CHANNEL_TO_EXTRACT = args.cth
SAMPLE_VECTOR_LENGTH = args.svs
DOWNSAMPLE_FACTOR = args.dsf

print('Filename to save: ', FILENAME_TO_SAVE)
print('Filename to process: ', FOLDER_TO_READ)
print('Channel no to extract: ', CHANNEL_TO_EXTRACT)
print('sample vector size: ', SAMPLE_VECTOR_LENGTH)
print('Downsample factor: ', DOWNSAMPLE_FACTOR)