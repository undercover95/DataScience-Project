# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import preprocess_emails as pe


def merge_data(spam, ham):
    spam_data = {
        'clean_msg': spam,
        'target': np.ones(len(spam))*(-1)
    }
    spam_data_df = pd.DataFrame(data=spam_data)

    ham_data = {
        'clean_msg': ham,
        'target': np.ones(len(ham))
    }
    ham_data_df = pd.DataFrame(data=ham_data)

    data_df = pd.concat([spam_data_df, ham_data_df])
    data_df = data_df.reset_index(drop=True)

    return data_df


spam, ham = pe.load_data_files(path='Data/*gz')

spam = pe.decode_emails(spam)
ham = pe.decode_emails(ham)
# end of preprocessing data

data_df = merge_data(spam, ham)

# np.save('spam.npy',spam)
# np.save('spam_sub.npy',spam_subjects)
# np.save('ham.npy',ham)
# np.save('ham_sub.npy',ham_subjects)

print(data_df)
data_df.to_pickle('email_cleaned')
# pd.read_pickle(file_name)
