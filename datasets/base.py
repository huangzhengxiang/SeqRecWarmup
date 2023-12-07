from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split
        self.split_timestamp = args.split_timestamp

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    # 定义一个函数来处理每个分组
    def process_group(self,group):
        # 按照timestamp进行排序
        sorted_group = group.sort_values(by='timestamp')
        # 找到时间戳大于self.split_timestamp部分
        gt_larger = sorted_group[sorted_group['timestamp'] >= self.split_timestamp]
        gt_smaller = sorted_group[sorted_group['timestamp'] < self.split_timestamp]
        # 如果大于self.split_timestamp的部分存在，则保留时间戳小于self.split_timestamp的部分
        # 和大于等于split_timestamp且最小的那个数据
        if not (gt_larger.empty or gt_smaller.empty) and len(gt_larger)>=self.args.leaven and len(gt_smaller)>=3:
            result = pd.concat([sorted_group[sorted_group['timestamp'] < self.split_timestamp], gt_larger.head(self.args.leaven)])
        elif (gt_larger.empty or len(gt_larger)<self.args.leaven) and len(sorted_group['timestamp'])>self.args.leaven and len(gt_smaller)>=3:
            result = pd.concat([sorted_group[sorted_group['timestamp'] < self.split_timestamp], sorted_group[sorted_group['timestamp'] >= self.split_timestamp]])
            # 如果大于等于self.split_timestamp的部分不存在，则不保留这个组
        else:
            result = pd.DataFrame()
        return result

    def make_split_timestamp(self,df):
        if self.split_timestamp!=0:
            grouped_df = df.groupby('uid')
            # 对每个分组应用处理函数
            result_df = grouped_df.apply(self.process_group)
            # 重置索引以得到最终结果
            df = result_df.reset_index(drop=True)
        return df

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.make_implicit(df)

        df = self.make_split_timestamp(df)
        # test 
        # grouped_df = df.groupby('uid')
        # validation_result = grouped_df.apply(lambda group: (group['timestamp'] >= self.split_timestamp).sum() == self.args.leaven)
        # print(validation_result)
        # assert False
        # df.to_csv("data.csv")
        # assert False
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw file doesn't exist. Downloading...")
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    def make_implicit(self, df):
        print('Turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating]
        # return df[['uid', 'sid', 'timestamp']]
        return df

    def filter_triplets(self, df: pd.DataFrame):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df

    def densify_index(self, df: pd.DataFrame):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']))}
        smap = {s: i + 1 for i, s in enumerate(set(df['sid']))} # items are 1-indexed
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                # train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
                # 将留1test扩展到留n，valid仍旧是一个
                train[user], val[user], test[user] = items[:-self.args.leaven-1], items[-self.args.leaven-1:-self.args.leaven], items[-self.args.leaven:]
            return train, val, test
        elif self.args.split == 'holdout':
            print('Splitting')
            np.random.seed(self.args.dataset_split_seed)
            eval_set_size = self.args.eval_set_size

            # Generate user indices
            permuted_index = np.random.permutation(user_count)
            train_user_index = permuted_index[                :-2*eval_set_size]
            val_user_index   = permuted_index[-2*eval_set_size:  -eval_set_size]
            test_user_index  = permuted_index[  -eval_set_size:                ]

            # Split DataFrames
            train_df = df.loc[df['uid'].isin(train_user_index)]
            val_df   = df.loc[df['uid'].isin(val_user_index)]
            test_df  = df.loc[df['uid'].isin(test_user_index)]

            # DataFrame to dict => {uid : list of sid's}
            train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            val   = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            test  = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            return train, val, test
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        if self.args.split == 'leave_one_out':
            folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}-timestamp{}-testsize{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split, self.split_timestamp,self.args.leaven)
        else:
            folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}-timestamp{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split, self.split_timestamp)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

