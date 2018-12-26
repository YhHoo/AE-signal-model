import argparse

print('Running script 2')

parser = argparse.ArgumentParser(description='Input some parameters.')
parser.add_argument('--sen', metavar='S', default=1, type=int, help='Number of Max Sent (Default: 3)')
parser.add_argument('--wrd', metavar='W', default=27, type=int, help='Number of Max Word (Default: 30)')

args = parser.parse_args()
MAX_SENTS = args.sen
MAX_SENT_LENGTH = args.wrd

print('sen: ', MAX_SENTS)
print('wrd: ', MAX_SENT_LENGTH)

print('Completed script 2')