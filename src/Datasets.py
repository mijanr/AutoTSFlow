from tsai.basics import *
from tslearn.preprocessing import TimeSeriesResampler 
from typing import Tuple
from sklearn.model_selection import train_test_split

class Datasets:
    def __init__(self) -> None:
        pass

    def get_data(self, dataset_name:str, ts_length:int, normalize:bool)->Tuple[np.ndarray, np.ndarray, list, dict]:
        """
        Get UCR_UEA datasets
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        ts_length : int or None
            Length of the time series (interpolation or truncation). 
            If None, the original time series length is used.
        normalize : bool
            If True, the time series are normalized (min-max scaling).

        Returns
        -------
        X : np.ndarray
            Time series data
        y : np.ndarray
            Target labels
        splits : list
            List of indices for train, validation splits
        test_data : dict
            Dictionary containing test data
            {'X': np.ndarray, 'y': np.ndarray}

        """
        X_train, y_train, X_test, y_test = get_classification_data(dataset_name, split_data=True)

        #first reshape to (n_ts, ts_len, d)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

        if ts_length is not None:
            X_train = TimeSeriesResampler(sz=ts_length).fit_transform(X_train)
            X_test = TimeSeriesResampler(sz=ts_length).fit_transform(X_test)
            
        if normalize:
            #normalize
            X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
            X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
        
        #then reshape back to (n_ts, d, ts_len)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

        #from train, keep 15% for validation
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
        #train/val split in tsai format
        X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])
        test_data = {'X': X_test, 'y': y_test}
        return X, y, splits, test_data
    
if __name__ == "__main__":
    ds = Datasets()
    X, y, splits, test_data = ds.get_data('Handwriting', ts_length=24, normalize=True)
    print(X.shape)
    print(y.shape)
    print(splits)
    print(test_data['X'].shape)
    print(test_data['y'].shape)
