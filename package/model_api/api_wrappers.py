import requests
import pandas as pd
import json
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os


class APIWrapper:
    def __init__(self):
        self.url = "https://colorado.posit.co/rsc/electronics-classifier/"
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

    def upload_data(self, df: pd.DataFrame) -> None:
        """
        Uploads data to the API.

        :param df: A DataFrame containing the data to be uploaded. Must contain 'text', 'annotator', and 'annotation' columns.
        :return: None
        """
        if not set(["text", "annotator", "annotation"]).issubset(df.columns):
            raise ValueError(
                "DataFrame must contain 'text', 'annotator', and 'annotation' columns."
            )
        data = df[["text", "annotator", "annotation"]].to_dict("records")
        return self.post("append_training_data", json=data)

    def query_data(self, qry: str) -> pd.DataFrame:
        """
        Queries data from the API.

        :param qry: A SQL query string.
        :return: A DataFrame containing the queried data.
        """
        json_response = self.get("query_data", params={"qry": qry})
        return pd.DataFrame(json.loads(json_response))

    def upload_model(self, model: XGBClassifier, vectorizer: CountVectorizer) -> str:
        """
        Uploads a model and a vectorizer to the API.

        :param model: An XGBClassifier model.
        :param vectorizer: A CountVectorizer.
        :return: The response from the API.
        """
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
        """
        Scores a model using the API.

        :param text: A string of text to be scored.
        :return: A number representing the probability that the text string is about electronics.
        """
        return self.get("score_model", params={"text": text})

    def last_updated(self) -> str:
        """
        Gets the last updated time from the API.

        :return: The last updated time as a string.
        """
        return self.get("last_updated")
