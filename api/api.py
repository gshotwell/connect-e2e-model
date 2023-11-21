from fastapi import FastAPI, UploadFile, Header, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Annotated, Optional
import duckdb
from datetime import datetime
import json
import uuid

app = FastAPI()

db_path = "training_data.db"

con = duckdb.connect(db_path)
con.execute(
    """
    CREATE TABLE IF NOT EXISTS training_data (
        id VARCHAR, 
        date TIMESTAMP,
        text VARCHAR, 
        annotator VARCHAR, 
        annotation BOOLEAN, 
    )
    """
)
con.close()

data_team = ["gordon.shotwell"]


class TrainingData(BaseModel):
    text: str
    annotator: str
    annotation: bool


class ModelMetadata(BaseModel):
    date: datetime
    model_name: str
    model_author: str


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


@app.post("/append_training_data")
async def append_training_data(
    data: List[TrainingData], user=Depends(get_current_user)
) -> Dict:
    validate_access(user, data_team)
    con = duckdb.connect(db_path)

    for item in data:
        id = uuid.uuid4
        date = datetime.now()
        query = (
            "INSERT INTO training_data (id, date, text, annotator, annotation) "
            f"VALUES ('{id}', '{date}', '{item.text}', '{item.annotator}', {item.annotation})"
        )
        con.execute(query)
    con.close()

    return {"number_of_added_entries": len(data)}


@app.get("/score_model")
async def score_model(text: str) -> int:
    pass


@app.put("/update_model")
async def update_model(
    model_file: UploadFile,
    metadata: ModelMetadata,
    user=Depends(get_current_user),
) -> None:
    validate_access(user, data_team)
    with open("model.bin", "wb") as f:
        f.write(await model_file.read())

    with open("model-metadata.json", "w") as f:
        json.dump(metadata.model_dump(), f)


@app.get("/model_metadata")
async def model_metadata() -> dict:
    with open("model-metadata.json", "r") as f:
        return json.load(f)


@app.get("/query_data")
async def query_data(qry: str, user=Depends(get_current_user)) -> List[TrainingData]:
    validate_access(user, data_team)
    con = duckdb.connect(db_path, read_only=True)
    result = con.execute(qry).fetchall()
    con.close()
    return [
        TrainingData(text=row[0], annotator=row[1], annotation=row[2]) for row in result
    ]


@app.get("/api_test")
async def api_test(qry: str) -> str:
    return qry


def validate_access(user: Optional[str], control_list: List) -> None:
    if user is None:
        return
    if user.user not in control_list:
        raise HTTPException(
            status_code=401, detail="You are not authorized to access this endpoint"
        )
