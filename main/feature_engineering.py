#feature_engineering.py

# Import the required libraries
import pandas as pd
import torch
import random
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import joblib
from .model import SparseRowDataset
from torch.utils.data import DataLoader
from .constants import TRAIN_DATA, TEST_DATA, DL_BATCH_SIZE

def get_file_size(path: str, encoding: str, hasHeader: bool =True):
    """ 
        File size for a provided path
    """
    total_lines = 0
    try:
        with open(path, 'r', encoding=encoding) as f:
            total_lines = sum(1 for line in f)
    except Exception as e:
        print(f"error in reading file: {e}")
    if hasHeader:
        total_lines -= 1
    return total_lines



def df_from_csv(path: str,
                nrows: int,
                *,
                seed: int | None = None,
                chunksize: int = 10_000,
                **read_kwargs) -> pd.DataFrame:
    """ 
        Return csv file contents as pandas dataframe
    """
    read_kwargs.setdefault("low_memory", False)
    if nrows < 1:
        return pd.read_csv(path, **read_kwargs)

    seeded = np.random.default_rng(seed)
    read_kwargs.setdefault("chunksize", chunksize)
    
    sampled = []
    skipped = 0
    col_names = None

    for chunk in pd.read_csv(path, **read_kwargs):
        if col_names is None:
            col_names = chunk.columns

        values = chunk.to_numpy()
        for row in values:
            skipped += 1
            if len(sampled) < nrows:
                sampled.append(row)
            else:
                curr = seeded.integers(skipped)
                if curr < nrows:
                    sampled[curr] = row
    return pd.DataFrame(sampled, columns=col_names)

def get_test_data():
    return df_from_csv(TEST_DATA)

def merge_df(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """ 
        Pre-processing of train and test dataframe
    """
    text_columns = [
        'name', 
        'username', 
        'mail_domain', 
        'subject', 
        'reply_to', 
        'x_mailer',
        'return_path', 
        'received', 
        'body_text', 
        'url', 
        'url_domain', 
        'url_path', 
        'url_query'
    ]

    numeric_columns = [
    'body_count_of_words', 
    'body_length', 
    'has_url', 
    'url_count',
    'has_attachment', 
    'attachment_count'
    ]

    for df in [train_df, test_df]:
        df[text_columns] = df[text_columns].fillna("")
        df[numeric_columns] = df[numeric_columns].fillna(0)
    return df

def df_col_aslist(df: pd.DataFrame, col: str) -> list[str]:
    return df[col].astype(str).tolist()

def _fit_transform(column, vectItem, train, test):
    if isinstance(vectItem, HashingVectorizer):
        _train = vectItem.transform(train)
        _test = vectItem.transform(test)
    else:
        _train = vectItem.fit_transform(train)
        _test = vectItem.transform(test)
    return \
        csr_matrix(_train, copy=False).astype(np.float32), \
        csr_matrix(_test, copy=False).astype(np.float32)

def featurize(train_df: pd.DataFrame, test_df: pd.DataFrame):
    defaults = dict(
        ngram_range = (1, 2),
        min_df = 2,
        stop_words = "english",
        sublinear_tf = True,
        dtype = np.float32
    )
    hashed_defaults = dict(
        ngram_range = (1, 2),
        norm = "l2",
        n_features = 3_000,        
        stop_words = "english",
        alternate_sign = False,
        dtype = np.float32
    )
    hash_column_candidates = [
        "mail_domain", 
        "reply_to", 
        "return_path", 
        "url_domain"
    ]
    # Column specific feature crafting
    vectorizers = {
        "name"        : TfidfVectorizer(max_features=300,  **defaults),
        "username"    : TfidfVectorizer(max_features=500,  **defaults),
        "mail_domain" : None,
        "subject"     : TfidfVectorizer(max_features=8_000, **defaults),
        "reply_to"    : None,
        "x_mailer"    : TfidfVectorizer(max_features=1_000, **defaults),
        "return_path" : None,
        "received"    : TfidfVectorizer(max_features=10_000, **defaults),
        "body_text"   : TfidfVectorizer(max_features=40_000, **defaults),
        "url"         : TfidfVectorizer(max_features=15_000, **defaults),
        "url_domain"  : None,
        "url_path"    : TfidfVectorizer(max_features=8_000, **defaults),
        "url_query"   : TfidfVectorizer(max_features=10_000, **defaults),
    }
    # Define fields for Hashing Vectorizers
    for col in hash_column_candidates:
        vectorizers[col] = HashingVectorizer(**hashed_defaults)

    # train & test corpora     
    text_columns = list(vectorizers.keys())
    train_corpora = {col: df_col_aslist(train_df, col) for col in text_columns}
    test_corpora = {col: df_col_aslist(test_df, col) for col in text_columns}

    with parallel_backend("loky", inner_max_num_threads=1):
        aggregate = Parallel(n_jobs=-1, verbose=4)(
            delayed(_fit_transform)(c, v, train_corpora[c], test_corpora[c])
            for c, v in vectorizers.items()
        )
    train, test = zip(*aggregate)
    X_train = hstack(train, format="csr", dtype=np.float32)
    X_test  = hstack(test,  format="csr", dtype=np.float32) 
    y_train = train_df['mail_is_safe'].astype(int).values
    y_test = test_df['mail_is_safe'].astype(int).values
    
    train_srd = SparseRowDataset(X_train, y_train)
    test_srd = SparseRowDataset(X_test, y_test)

    return train_srd, test_srd

def get_dataloader(df: pd.DataFrame, 
                   batch_size: int = DL_BATCH_SIZE, 
                   shuffle: bool = False, 
                   drop_last: bool = False) -> DataLoader:
    return DataLoader(df, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def imbalance_ratio(train_loader):
    labels = torch.cat([yb for _, yb in train_loader])
    neg = (labels == 0).sum().item()
    pos = (labels == 1).sum().item()
    return neg / max(pos, 1) 

def process(nrows: int=-1, pathfix: str=''):
    train_df = df_from_csv(pathfix+TRAIN_DATA, nrows)
    test_df = df_from_csv(pathfix+TEST_DATA, nrows)
    df = merge_df(train_df, test_df)
    train, test = featurize(train_df, test_df)
    train_loader = get_dataloader(df=train, shuffle=True, drop_last=True)
    test_loader = get_dataloader(test)
    return train_loader, test_loader

if __name__ == '__main__':
    process()