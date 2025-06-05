# python 3
# This script is modified from the script by Kuparinen and Scherrer (2024) to fit GCND
# It requires all texts/ documents in one single folder, in .txt format
# Last update: June 2025

import sys
import os
import glob
import json
import re
from nltk import ngrams

corpus = sys.argv[1]

# Find files
joined_files = os.path.join("*txt")
joined_list = glob.glob(joined_files)

# define tussenwerpsels from GCND annotation manual (March 2024)
tussenwerpsels = ["ah","aha","ah ja", "awel", "allez",
                  "ai","au","bah","boe","bè","bè ja",
                  "dè","ei","eikes","eneeë","eni",
                  "ewaar","goh","ha","haha","hé",
                  "hè","hei","ho","hu","hum","ja","jee",
                  "o jee","mm-hu","mmm","moh","mordjie",
                  "neeë","nou","oeh","oei","oesje","oh",
                  "o","oho","poeh","pst","sjt","sst","tè",
                  "tut","uh","uhm","uhu","wauw","who","zuh",
                  "zulle","zunne","zun","wi","zi","ggg","xxx"]

# Bring to corpus
import codecs
files = [codecs.open(file, "r", "utf-8").read() for file in joined_list]
words = [re.sub(r'[^\w#]+', ' ', sent) for sent in files] # allowing '#' for GCND
words = [file.split() for file in words] # split text into list of words
words = [list(element.lower() for element in file if element not in tussenwerpsels) for file in words ]
words = [' '.join(file) for file in words]
# print(words)

with open('words_{}'.format(corpus), 'w', encoding ='utf-8') as json_file:
    json.dump(words, json_file, ensure_ascii = False)

with open("./GCND_full/words_gcnd_full_modified_cleaned", "r", encoding="utf-8") as fp:
    words = json.load(fp)

lines = [re.sub(r' ', '_ _', sent) for sent in words]
lines = ["_" + file for file in lines]
lines = [file + "_" for file in lines]


# # Split to trigrams
trigram = [["".join(k1).lower() for k1 in list(ngrams(file, 3))] for file in lines]
trigram = [[re.sub(r'(\w*\s\w*)', '', element) for element in file] for file in trigram]
trigram = [str([item for item in sublist if len(item) == 3]) for sublist in trigram]
# print(trigram)

# Split to fourgrams
fourgram = [["".join(k1) for k1 in list(ngrams(file, 4))] for file in lines]
fourgram = [[re.sub(r'(\w*\s\w*)', '', element) for element in file] for file in fourgram]
fourgram = [str([item for item in sublist if len(item) == 4]) for sublist in fourgram]
# print(fourgram)

with open('trigram_{}'.format(corpus), 'w', encoding ='utf8') as json_file:
    json.dump(trigram, json_file, ensure_ascii = False)

with open('fourgram_{}'.format(corpus), 'w', encoding ='utf8') as json_file:
    json.dump(fourgram, json_file, ensure_ascii = False)