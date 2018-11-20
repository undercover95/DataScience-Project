# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:50:35 2018

@author: portable Otto
"""

import psutil
import glob
import tarfile
import pandas as pd
import numpy as np
import tqdm
import string
import unicodedata
import sys

tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

# returns the percentile of used ram, using as a break to not shutdown everything
psutil.virtual_memory().percent

path1 = 'Data/*gz'
files = glob.glob(path1)

data = []


ham = []
spam = []

for name in files[2:]:
    tar = tarfile.open(name, "r:gz")
    for member in tar.getmembers():
        if('ham' in member.name):
            f = tar.extractfile(member)
            if f is not None:
                ham.append(f.read())
        elif('spam' in member.name):
            f = tar.extractfile(member)
            if f is not None:
                spam.append(f.read())


subject = b"\n"
spam_subjects = []

to_del = []

for i in tqdm.tqdm(range(len(spam))):
    index = spam[i].find(subject)
    # from index 9 to get rid of 'Subject: '
    spam_subjects.append(spam[i][9:index])
    text = spam[i][index + len(subject):]
    if text != b'':
        try:
            spam[i] = set(text.decode("utf-8").replace("\n",
                                                       " ").replace("\r", " ").translate(tbl).lower())
        except UnicodeDecodeError:
            to_del.append(i)
    else:
        spam[i] = ''

for item in reversed(to_del):  # removing all emails that couldnt be decoded
    del spam[item]
    del spam_subjects[item]

to_del = []

ham_subjects = []
for i in tqdm.tqdm(range(len(ham))):
    index = ham[i].find(subject)
    # from index 9 to get rid of 'Subject: '
    ham_subjects.append(ham[i][9:index])
    text = ham[i][index + len(subject):]
    if text != b'':
        try:
            ham[i] = set(text.decode("utf-8").replace("\n",
                                                      " ").replace("\r", " ").translate(tbl).lower().split(' '))
        except UnicodeDecodeError:
            to_del.append(i)
    else:
        ham[i] = ''

for item in reversed(to_del):  # removing all emails that couldnt be decoded
    del spam[item]
    del spam_subjects[item]


# end of preprocessing data
print(ham[0])

# np.save('spam.npy',spam)
# np.save('spam_sub.npy',spam_subjects)
# np.save('ham.npy',ham)
# np.save('ham_sub.npy',ham_subjects)

# df.to_pickle(file_name)
# pd.read_pickle(file_name)
