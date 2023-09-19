from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

EVAL_FUNCTIONS = {
    "token_ratio": fuzz.token_ratio,
    "ratio": fuzz.ratio,
    "partial_token_ratio": fuzz.partial_token_ratio,
    "partial_ratio": fuzz.partial_ratio,
}


def _join_df_content(df, joiner="\t"):
    """joining dataframe's table content as one long string"""
    return joiner.join(df.values.ravel())


def default_tokenizer(text: str) -> List[str]:
    """a simple tokenizer that splits text by white space"""
    return text.split()


def compare_contents_as_df(
    actual_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    joiner: str = "\t",
    eval_func: str = "token_ratio",
    processor: Callable = None,
):
    """ravel the table as string then use text distance to compare the prediction against true
    table

    Parameters
    ----------
    actual_df: pd.DataFrame
        actual table as pandas dataframe

    pred_df: pd.DataFrame
        predicted table as pandas dataframe

    joiner: str, default to "\t"
        the string to join cells together

    eval_func: str, default tp "token_ratio"
        the eval_func should be one of "token_ratio", "ratio", "partial_token_ratio",
        "partial_ratio". Those are functions provided by rapidfuzz to evaluate text distances
        using either tokens or characters. In general token is better than characters for evaluating
        tables.

    processor: Callable, default to None
        processor to tokenize the text; by default None means no processing (using characters). For
        tokens eval functions we recommend using the `default_tokenizer` or some other functions to
        break down the text into words

    """
    func = EVAL_FUNCTIONS.get(eval_func)
    if func is None:
        raise ValueError(
            'eval_func must be one of "token_ratio", "ratio", "partial_token_ratio", '
            f'"partial_ratio" but got {eval_func}',
        )
    return {
        f"by_col_{eval_func}": func(
            _join_df_content(actual_df),
            _join_df_content(pred_df),
            processor=processor,
        ),
        f"by_row_{eval_func}": func(
            _join_df_content(actual_df.T),
            _join_df_content(pred_df.T),
            processor=processor,
        ),
    }
