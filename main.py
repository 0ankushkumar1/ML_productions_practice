# main.py
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import features  # noqa: F401 — lets pickle resolve MovieFeatureEngineer

app = FastAPI(title="IMDb Rating Predictor")

with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)
# after loading the pipeline
feature_engineer = pipeline.named_steps["features"]
model = pipeline.named_steps["model"]
EXPECTED_COLS = list(model.feature_names_in_)


class Movie(BaseModel):
    title: str = ""
    release_date: str = None
    vote_average: float = None
    vote_count: float = None
    popularity: float = None
    budget: float = None
    revenue: float = None
    runtime: float = None
    genres: str = ""
    keywords: str = ""
    cast: str = ""
    production_companies: str = ""
    production_countries: str = ""
    spoken_languages: str = ""
    director: str = None
    writers: str = ""
    producers: str = ""
    overview: str = ""
    tagline: str = ""


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(movie: Movie):
    df = pd.DataFrame([movie.dict()])
    transformed = feature_engineer.transform(df)
    transformed = transformed[EXPECTED_COLS]   # force exact training order
    pred = float(model.predict(transformed)[0])
    return {"predicted_imdb_rating": round(min(max(pred, 0), 10), 2)}