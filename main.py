import hydra
from omegaconf import DictConfig
import numpy as np

@hydra.main(version_base='1.3.2', config_path="configs", config_name="main_config.yaml")
def main(cfg: DictConfig) -> None:
    dataset = hydra.utils.instantiate(cfg.datasets.model_params)

    #load the data
    X, y, splits, test_data = dataset.get_data(dataset_name = cfg.dataset_name[0], **cfg.datasets.data_params)

    #data params
    data_params = {
        "c_in": X.shape[1],
        "c_out": len(np.unique(y)),
        "seq_len": X.shape[2]
    }

    model = hydra.utils.instantiate(cfg.models.model, **data_params)

if __name__ == "__main__":
    main()
