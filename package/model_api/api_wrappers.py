os.getenv('CONNECT_API_KEY')}


import requests
import pandas as pd
import json
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os


class APIWrapper:
    def __init__(self):
        self.url = "http://127.0.0.1:8000/"
        self.headers = {"Authorization": f"Key {os.getenv('CONNECT_API_KEY')}"}

    def post(self, endpoint: str, **kwargs) -> requests.Response:
        resp = requests.post(self.url + endpoint, headers=self.headers, **kwargs)
        if resp.status_code == 401:
            raise Exception("You don't have access to this endpoint")
        return resp

    def get(self, endpoint: str, **kwargs) -> requests.Response:
        resp = requests.get(self.url + endpoint, headers=self.headers, **kwargs)
        if resp.status_code == 401:
            raise Exception("You don't have access to this endpoint")
        if resp.status_code == 200:
            return resp.content.decode()
        return resp

    def test(self) -> str:
        return self.get("api_test", params={"qry": "Test sucessful!"})

    def upload_data(self, df: pd.DataFrame) -> None:
        if not set(["text", "annotator", "annotation"]).issubset(df.columns):
            raise ValueError(
                "DataFrame must contain 'text', 'annotator', and 'annotation' columns."
            )
        data = df[["text", "annotator", "annotation"]].to_dict("records")
        return self.post("append_training_data", json=data)

    def query_data(self, qry: str) -> pd.DataFrame:
        json_response = self.get("query_data", params={"qry": qry})
        return pd.DataFrame(json.loads(json_response))

    def upload_model(self, model: XGBClassifier, vectorizer: CountVectorizer) -> str:
        import pickle

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        files = {
            "ml_model": open("model.pkl", "rb"),
            "vectorizer": open("vectorizer.pkl", "rb"),
        }
        return self.post("update_model", files=files)

    def score_model(self, text: str) -> float:
        return self.get("score_model", params={"text": text})

    def last_updated(self) -> str:
        return self.get("last_updated")
