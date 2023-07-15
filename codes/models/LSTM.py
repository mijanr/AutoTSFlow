from tsai.all import *
import fastai
from tsai.models.RNN import LSTM
import numpy as np

class LSTM_classifier:
    """
    LSTM classifier for time series classification

    Parameters
    ----------
    c_in: int
        Number of input channels
    c_out: int
        Number of classes
    hidden_size: int
        Number of hidden units in LSTM
    n_layers: int
        Number of LSTM layers
    bias: bool
        Whether to use bias in LSTM
    rnn_dropout: float
        Dropout rate of LSTM output
    bidirectional: bool
        Whether to use bidirectional LSTM
    fc_dropout: float
        Dropout rate of FC layer
    init_weights: bool
        Whether to initialize the weights   

    """

    def __init__(
            self, 
            c_in,
            c_out,
            hidden_size=128,
            n_layers=1,
            bias=True,
            rnn_dropout=0.0,
            bidirectional=False,
            fc_dropout=0.0,
            init_weights=True,
            **kwargs
    ):
        self.model = LSTM(
            c_in = c_in,
            c_out = c_out,
            hidden_size = hidden_size,
            n_layers = n_layers,
            bias = bias,
            rnn_dropout = rnn_dropout,
            bidirectional = bidirectional,
            fc_dropout = fc_dropout,
            init_weights = init_weights
        )

    def fit(
            self, 
            X: np.ndarray,
            y: np.ndarray,
            splits: list,
            test_data: dict,
            epochs: int,
            lr: float
    ):
        """
        Fit the model on the training data

        Parameters
        ----------
        X: np.ndarray
            Input data
        y: np.ndarray
            Target data
        splits: list
            List of indices for training and validation splits
        test_data: dict
            Dictionary of test data
            {'X': X_test, 'y': y_test}
        epochs: int
            Number of epochs to train
        lr: float
            Learning rate
        """
        tfms = [None, [Categorize()]]
        dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)
        learn = Learner(dls, self.model, metrics=fastai.metrics.accuracy)
        learn.fit_one_cycle(epochs, lr)

        #now add the test data to the dls
        test_ds = dls.valid.dataset.add_test(**test_data)
        test_dl = dls.valid.new(test_ds)

        #get the predictions
        _, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True)
        accuracy = skm.accuracy_score(test_targets, test_preds)
        cl_report = skm.classification_report(test_targets, test_preds, output_dict=True)
        return {"accuracy": accuracy, "cl_report": cl_report, "target": test_targets, "pred": test_preds}
    

if __name__ == "__main__":
    clf = LSTM_classifier(
        c_in = 1,
        c_out = 2
    )
    print(clf)