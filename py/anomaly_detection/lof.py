import os
import pickle
import numpy as np

from dataclasses import dataclass
from pyod.models.lof import LOF

import logging


@dataclass
class CustomParameters:
    window_size: int = 100
    n_neighbors: int = 20
    leaf_size: int = 30
    distance_metric_order: int = 2
    n_jobs: int = 1
    algorithm: str = "auto"  # using default is fine
    distance_metric: str = "minkowski"  # using default is fine
    random_state: int = 123
    use_column_index: int = 0


def set_random_state(customParameters: CustomParameters) -> None:
    seed = customParameters.random_state
    import random

    random.seed(seed)
    np.random.seed(seed)


def lof(data, sensor, args, data_type):
    ad_args = args["detect_anomalies"]
    model_dir = ad_args["directory"] or "models"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    filename = f'{model_dir}/{sensor["source"]}.{data_type}.pickle'

    if ad_args["use_existing_model"]:
        with open(filename, "rb") as file:
            model = pickle.load(file)

        new_scores = model.decision_function(data)
        previous_scores = model.decision_scores_

        anomaly_bound = get_anomaly_bound(previous_scores)
        new_labels = (new_scores > anomaly_bound).astype(int)

        return new_scores, new_labels

    else:
        n_neighbors = int(np.ceil(0.3 * len(data)))

        logging.info(
            f"{sensor['name']}: Detecting anomalies with lof using {n_neighbors} for n_neighbors"
        )

        customParameters = CustomParameters(n_neighbors=n_neighbors)

        set_random_state(customParameters)

        model = LOF(
            contamination=np.nextafter(0, 1),
            n_neighbors=customParameters.n_neighbors,
            leaf_size=customParameters.leaf_size,
            n_jobs=customParameters.n_jobs,
            algorithm=customParameters.algorithm,
            metric=customParameters.distance_metric,
            metric_params=None,
            p=customParameters.distance_metric_order,
        )
        model.fit(data)

        with open(filename, "wb") as file:
            pickle.dump(model, file)
        scores = model.decision_scores_

        anomaly_bound = get_anomaly_bound(scores)
        labels = (scores > anomaly_bound).astype(int)

        return scores, labels


def get_anomaly_bound(scores):
    iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
    anomaly_bound = np.percentile(scores, 75) + 1.0 * iqr  # 1.5 * iqr

    return anomaly_bound
