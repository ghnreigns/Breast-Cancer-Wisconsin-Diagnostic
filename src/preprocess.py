from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import base
import numpy as np
from typing import Union, List
import pandas as pd


class ReduceVIF(base.BaseEstimator, base.TransformerMixin):
    """The base of the class structure is not implemented by me, however, I heavily modified the class such that it can
    take in numpy arrays and correctly implemented the fit and transform method.
    """

    def __init__(self, thresh=10):
        self.thresh = thresh
        self.feature_names_ = None
        self.predictor_cols = [
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
        ]

    def reset(self):
        """Resets the state of predictor columns after each fold."""

        self.predictor_cols = [
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
        ]

    def fit(self, X, y=None):
        """Fits the Recursive VIF on the training folds and save the selected feature names in self.feature_names

        Args:
            X ([type]): [description]
            y ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        # print("ReduceVIF fit")
        tmp, self.predictor_cols = ReduceVIF.calculate_vif(X, self.predictor_cols, self.thresh)
        self.feature_names_ = self.predictor_cols  # save as an attribute to call later
        col_index = [self.predictor_cols.index(col_name) for col_name in self.predictor_cols]
        self.col_index = col_index
        self.reset()
        return self

    def transform(self, X, y=None):
        """Transforms the Validation Set according to the selected feature names.

        Args:
            X ([type]): [description]
            y ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        # print("ReduceVIF transform")
        return X[:, self.col_index]

    @staticmethod
    def calculate_vif(X: Union[np.ndarray, pd.DataFrame], columns: List[str], thresh: float = 10.0):
        """Implements a VIF function that recursively eliminates features.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): [description]
            columns (List[str]): [description]
            thresh (float, optional): [description]. Defaults to 10.0.

        Returns:
            [type]: [description]
        """

        dropped = True
        count = 0
        while dropped and count <= 15:
            column_index = X.shape[1]
            predictor_cols = np.arange(X.shape[1])
            dropped = False
            vif = []
            for var in range(column_index):
                vif.append(variance_inflation_factor(X[:, predictor_cols], var))

            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                # print(f"Dropping {maxloc} with vif={max_vif}")
                # X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                X = np.delete(X, maxloc, axis=1)
                columns.pop(maxloc)
                dropped = True
                count += 1
        return X, columns
