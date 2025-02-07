from typing import Tuple
import pandas as pd
import os


class SoundDatasetFirstTime:
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset manager with the path to the data directory.
        :param data_dir: Path to the main 'data' directory.
        """
        self.dataset_path = dataset_path

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the data from the given data directory.
        :param data_splitting: Type of data to load (e.g. 'train', 'test', 'val').
        :return: A tuple of pandas DataFrames containing the loaded data for train, val, and test.
        """
        df_test = self.load_test()
        df_val = self.load_val()
        df_train = self.load_train()
        return df_train, df_val, df_test

    def load_train(self) -> pd.DataFrame:
        """
        Load the training data from the dataset.
        :return: A pandas DataFrame containing the training data.
        """
        data_label_path = os.path.join(self.dataset_path, "train", "train.csv")
        df_train = pd.read_csv(data_label_path)
        df_train = self.add_wav_path(df_train, "train")
        return df_train
    
    def load_val(self) -> pd.DataFrame:
        """
        Load the validation data from the dataset.
        :return: A pandas DataFrame containing the validation data.
        """
        data_label_path = os.path.join(self.dataset_path, "val/val.csv")
        df_val = pd.read_csv(data_label_path)
        df_val = self.add_wav_path(df_val, "val")
        return df_val
    
    def load_test(self) -> pd.DataFrame:
        """
        Load the test data from the dataset.
        :return: A pandas DataFrame containing the test data.
        """
        wav_files = [f for f in os.listdir(os.path.join(self.dataset_path, 'test')) if f.endswith('.wav')]
        wav_ids = [os.path.splitext(f)[0] for f in wav_files]
        df_test = pd.DataFrame(wav_ids, columns=['wav_id'])
        df_test = self.add_wav_path(df_test, "test")
        return df_test

    def add_wav_path(self, df, data_splitting: str):
        df['audio_path'] = df['wav_id'].apply(lambda x: f"{self.dataset_path}/{data_splitting}/{x}.wav")
        return df

class SoundDataset:
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset manager with the path to the data directory.
        :param data_dir: Path to the main 'data' directory.
        """
        self.dataset_path = dataset_path

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the data from the given data directory.
        :param data_splitting: Type of data to load (e.g. 'train', 'test', 'val').
        :return: A tuple of pandas DataFrames containing the loaded data for train, val, and test.
        """
        df_test = self.load_test()
        df_val = self.load_val()
        df_train = self.load_train()
        return df_train, df_val, df_test

    def load_train(self) -> pd.DataFrame:
        """
        Load the training data from the dataset.
        :return: A pandas DataFrame containing the training data.
        """
        data_label_path = os.path.join(self.dataset_path, "train_dataset.xlsx")
        df_train = pd.read_excel(data_label_path, index_col=False)
        return df_train
    
    def load_val(self) -> pd.DataFrame:
        """
        Load the validation data from the dataset.
        :return: A pandas DataFrame containing the validation data.
        """
        data_label_path = os.path.join(self.dataset_path, "val_dataset.xlsx")
        df_val = pd.read_excel(data_label_path, index_col=False)
        return df_val
    
    def load_test(self) -> pd.DataFrame:
        """
        Load the test data from the dataset.
        :return: A pandas DataFrame containing the test data.
        """
        data_label_path = os.path.join(self.dataset_path, "test_dataset.xlsx")
        df_test = pd.read_excel(data_label_path, index_col=False)
        return df_test

    def save_dataset(self, df, name):
        df.to_excel(f"../Dataset/{name}.xlsx", index=False)
        print(f"Dataset saved as {name}.xlsx")  

