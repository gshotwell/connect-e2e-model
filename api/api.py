from fastapi import FastAPI, UploadFile, Header, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Annotated, Optional
import pandas as pd
import json
import uuid
import sqlite3
import os
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pickle

parent = Path(__file__).parent
db_path = parent / "training_db.sqlite"


async def load_data():
    """
    Load data from the 'electronics.parquet' file into a SQLite database.
    If the database does not exist, it is created.
    """
    global db_path
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_parquet(parent / "electronics.parquet")
        df.to_sql("training_data", conn, if_exists="replace", index=False)
        conn.close()


app = FastAPI()

app.add_event_handler("startup", load_data)

data_team = ["gordon.shotwell"]


class TrainingData(BaseModel):
    """
    A Pydantic model for training data.
    """

    text: str
    annotator: str
    annotation: bool


class UserMetadata(BaseModel):
    """
    A Pydantic model for user metadata.
    """

    user: str
    groups: list[str] = Field(list)


async def get_current_user(
    rstudio_connect_credentials: Annotated[str | None, Header()] = None
) -> UserMetadata | None:
    """
    Get the user metadata from the RStudio-Connect-Credentials header and then
    parse the data into a UserMetadata object.
    """
    if rstudio_connect_credentials is None:
        return None
    user_meta_data = json.loads(rstudio_connect_credentials)
    return UserMetadata(**user_meta_data)


@app.post("/append_training_data")
async def append_training_data(
    data: List[TrainingData], user=Depends(get_current_user)
) -> Dict:
    """
    Append training data to the 'electronics.parquet' file.
    """
    validate_access(user, data_team)

    df = pd.DataFrame([item.dict() for item in data])
    df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["date"] = pd.Timestamp.now()
    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_path)
    df.to_sql("training_data", conn, if_exists="append", index=False)
    conn.close()

    return {"number_of_added_entries": len(data)}


@app.get("/score_model")
async def score_model(text: str) -> float:
    """
    Score a text using the model.
    """
    with open("model.bin", "rb") as f:
        model: XGBClassifier = pickle.load(f)
    with open("vectorizer.bin", "rb") as f:
        vectorizer: CountVectorizer = pickle.load(f)
    text_vector = vectorizer.transform([text])
    score = model.predict_proba(text_vector)
    return score[0][1]


@app.post("/update_model")
async def update_model(
    ml_model: UploadFile,
    vectorizer: UploadFile,
    user=Depends(get_current_user),
) -> None:
    """
    Update the model and vectorizer.
    """
    validate_access(user, data_team)
    with open("model.bin", "wb") as f:
        f.write(await ml_model.read())

    with open("vectorizer.bin", "wb") as f:
        f.write(await vectorizer.read())

    with open("last_updated.txt", "w") as f:
        f.write(str(pd.Timestamp.now()))


@app.get("/last_updated")
async def model_metadata() -> str:
    """
    Get the last updated time of the model.
    """
    with open("last_updated.txt", "r") as f:
        last_updated = f.read()
    return last_updated


@app.get("/query_data")
async def query_data(qry: str, user=Depends(get_current_user)) -> List[Dict]:
    """
    Query data from the SQLite database.
    """
    validate_access(user, data_team)
    global db_path
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(qry)
    result = cur.fetchall()
    column_names = [description[0] for description in cur.description]
    result = pd.DataFrame(result, columns=column_names)
    con.close()
    return result.to_dict("records")


@app.get("/api_test")
async def api_test(qry: str, user=Depends(get_current_user)) -> str:
    """
    A test endpoint that returns the input query string.
    """
    return user.user


def validate_access(user: Optional[str], control_list: List) -> None:
    """
    Validate the access of a user.
    If the user is not in the control list, an HTTPException is raised.
    """
    if user is None or user.user not in control_list:
        raise HTTPException(
            status_code=401, detail="You are not authorized to access this endpoint"
        )
