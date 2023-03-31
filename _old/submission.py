from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt


def generate_submission_file(
    filename: str,
    observation_ids: npt.ArrayLike,
    s_pred: list[Iterable],
) -> None:
    """Generate submission file for Kaggle

    Parameters
    ----------
    filename : string
        Submission filename.
    observation_ids : 1d array-like
        Test observations ids
    s_pred : list of 1d array-like
        Set predictions for test observations.
    """
    s_pred = [" ".join(map(str, pred_set)) for pred_set in s_pred]

    df = pd.DataFrame(
        {
            "Id": observation_ids,
            "Predicted": s_pred,
        }
    )
    df.to_csv(filename, index=False)
