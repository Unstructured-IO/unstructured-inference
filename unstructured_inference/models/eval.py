from typing import Dict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def eval_cells(actual_df: pd.DataFrame, predicted_df: pd.DataFrame) -> Dict[str, float]:
    """compare an actual table against a predicted table's content by mearning the corresponding
    cell's strings' similarity (rapidfuzz.fuzz.ratio)

    Returns
    -------
    np.ndarray
        shape the same as `actual_df`, [i, j] is the rapidfuzz.fuzz.ratio(actual_df[i, j],
        predicted_df[i, j]); if the predicted_df is smaller than actual_df's shape, the
        corresponding missing cell's score is 0; extra cells in predicted_df is not punished in this
        metric
    """
    scores = np.zeros(actual_df.shape)
    filled_pred = predicted_df.fillna("")
    pred_nrow, pred_ncol = predicted_df.shape
    for irow, row in actual_df.fillna("").iterrows():
        if irow >= pred_nrow:
            scores[icol, :] = 0
            continue

        pred_row = filled_pred.iloc[irow].values
        for icol, actual in enumerate(row.values):
            if icol >= pred_ncol:
                scores[irow, icol] = 0
            else:
                scores[irow, icol] = fuzz.ratio(actual, pred_row[icol])

    return scores