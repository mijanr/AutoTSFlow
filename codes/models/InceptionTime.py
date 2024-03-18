from tsai.all import *
import fastai
from tsai.models.InceptionTime import InceptionTime
import numpy as np

class InceptionTime_classifier:
    """
    InceptionTime classifier for time series classification

    Parameters  
    ----------
    c_in: int
        Number of input channels
    c_out: int
        Number of classes
    seq_len: int
        Length of the input sequence
    nf: int
        Number of filters
    nb_filters: int
        Number of filters in the inception module   
    ks: int
        Kernel size 
    bottleneck: bool
        Whether to use bottleneck

    Attributes
    ----------
    model: nn.Module
        InceptionTime model
    """

    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        nf=32,
        nb_filters=32,
        ks=40,
        bottleneck=True,
        **kwargs
    ):
        self.model = InceptionTime(
            c_in=c_in,
            c_out=c_out,
            seq_len=seq_len,
            nf=nf,
            nb_filters=nb_filters,
            ks=ks,
            bottleneck=bottleneck,
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
    # instantiate the model
    model = InceptionTime_classifier(c_in=1, c_out=2, seq_len=100)