from fastapi import FastAPI, UploadFile, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Annotated
import duckdb
from datetime import datetime
import json

app = FastAPI()

db_path = "training_data.db"

con = duckdb.connect(db_path)
con.execute(
    "CREATE TABLE IF NOT EXISTS training_data (text VARCHAR, annotator VARCHAR, annotation BOOLEAN)"
)
con.close()


class TrainingData(BaseModel):
    text: str
    annotator: str
    annotation: bool


class ModelMetadata(BaseModel):
    date: datetime
    model_name: str
    model_author: str


@app.post("/append_training_data")
async def append_training_data(data: List[TrainingData]) -> Dict:
    con = duckdb.connect(db_path)

    for item in data:
        query = f"INSERT INTO training_data (text, annotator, annotation) VALUES ('{item.text}', '{item.annotator}', {item.annotation})"
        con.execute(query)
    con.close()

    return {"number_of_added_entries": len(data)}


@app.get("/score_model")
async def score_model(text: str) -> int:
    pass


@app.put("/update_model")
async def update_model(model_file: UploadFile, metadata: ModelMetadata) -> None:
    with open("model.bin", "wb") as f:
        f.write(await model_file.read())

    with open("model-metadata.json", "w") as f:
        json.dump(metadata.model_dump(), f)


@app.get("/model_metadata")
async def model_metadata() -> dict:
    with open("model-metadata.json", "r") as f:
        return json.load(f)


@app.get("/query_data")
async def query_data(qry: str) -> List[TrainingData]:
    con = duckdb.connect(db_path, read_only=True)
    result = con.execute(qry).fetchall()
    con.close()
    return [
        TrainingData(text=row[0], annotator=row[1], annotation=row[2]) for row in result
    ]


class UserMetadata(BaseModel):
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


@app.get("/hello")
async def get_hello(user=Depends(get_current_user)) -> JSONResponse:
    """
    Use FastAPIs dependency injection system to get the current user.
    """
    if user is None:
        return {"message": "Howdy stranger!"}
    return {"message": f"So nice to see you, {user.user}!"}
