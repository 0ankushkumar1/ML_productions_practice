# features.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MovieFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        date = pd.to_datetime(X["release_date"], errors="coerce")
        X["release_year"] = date.dt.year
        X["release_month"] = date.dt.month
        X["release_quarter"] = date.dt.quarter
        X["release_dayofweek"] = date.dt.dayofweek

        X["title_length"] = X["title"].fillna("").astype(str).str.len()
        X["overview_length"] = X["overview"].fillna("").astype(str).str.len()
        X["tagline_length"] = X["tagline"].fillna("").astype(str).str.len()

        list_cols = ["genres", "keywords", "cast", "production_companies",
                     "production_countries", "spoken_languages", "writers", "producers"]
        for col in list_cols:
            X[f"{col}_count"] = (
                X[col].fillna("").astype(str)
                .apply(lambda x: len([v for v in x.split("|") if v.strip()]))
            )

        X["has_director"] = X["director"].notna().astype(int)

        for col in ["vote_count", "popularity", "budget", "revenue"]:
            X[f"log_{col}"] = np.log1p(X[col].clip(lower=0))

        for col in ["budget", "revenue", "runtime"]:
            X[f"has_{col}"] = X[col].notna().astype(int)

        X = X.drop(columns=[
            "id", "title", "status", "release_date", "genres", "keywords",
            "cast", "production_companies", "production_countries",
            "spoken_languages", "director", "writers", "producers",
            "overview", "tagline", "imdb_rating"
        ], errors="ignore")

        return X.replace([np.inf, -np.inf], np.nan)