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


def join_df_content(df, joiner="\t"):
    return joiner.join(df.values.ravel())


def compare_contents_as_cells(actual_cells, pred_cells):
    actual_df = table_cells_to_dataframe(actual_cells, 1, 1).fillna("")
    pred_df = table_cells_to_dataframe(pred_cells, 1, 1).fillna("")
    return compare_contents_as_df(actual_df, pred_df)


def compare_contents_as_df(actual_df, pred_df, joiner="\t"):
    """ravel the table as string then use text distance to compare the prediction against true
    table;

    When choosing joiner = "\t" (default) we evalute the structure as well as content (if a cell
    with two words is split into two cells there is an extra "\t")

    If choose joiner = " " (white space) we do not evaluate the structure but purely evalute if the
    text is in the right order. A cell with two words could have been split into two cells each with
    one word but the order still matches true table so it won't reduce the score
    """
    return {
        "by_col": fuzz.partial_ratio_alignment(
            join_df_content(actual_df), join_df_content(pred_df)
        ),
        "by_row": fuzz.partial_ratio_alignment(
            join_df_content(actual_df.T), join_df_content(pred_df.T)
        )
    }
