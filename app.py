from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
import os
from pydantic import BaseModel
from typing import List
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("catboost_model.cbm")  
    model = CatBoostClassifier() 
    model.load_model(model_path) 
    return model

app = FastAPI()

db = (
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)

def get_engine():
    return create_engine(db)



def batch_load_sql(query: str) -> pd.DataFrame:
    """
    На этом этапе загружаем данные из Postgres по чанкам, чтобы не тратить слишком много
    оперативной памяти за раз
    """


    CHUNKSIZE = 200000

    engine = get_engine()
    conn = engine.connect().execution_options(stream_results=True)

    chunks = []

    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    
    conn.close()

    if not chunks:
        return pd.DataFrame()
    

    return pd.concat(chunks, ignore_index=True)

def load_user_features():
    """
    На этом этапе читаем таблицу с колонокой user_id из БД с фичами пользователей
    """

    TABLE_NAME = "arsenij_shmelev_gwh6987_features_lesson_22_user"  
    query = f"SELECT * FROM {TABLE_NAME}"

    user_features = batch_load_sql(query)

    user_features = user_features.set_index("user_id")


    return user_features



def load_post_features():
    """
    На этом этапе читаем таблицу с колонокой user_id из БД с фичами пользователей
    """

    TABLE_NAME = "arsenij_shmelev_gwh6987_features_lesson_22_post"

    engine = get_engine()

    post_features = pd.read_sql(
        f"SELECT * FROM public.{TABLE_NAME}",
        con=engine
    )

    post_features = post_features.set_index("post_id")

    post_text_df = pd.read_sql(f"SELECT post_id, text, topic FROM public.post_text_df", con=engine)

    posts_df = post_text_df.rename(columns={"post_id": "id"}).set_index("id")

    return post_features, posts_df
    



model = load_models()                     
user_features = load_user_features()      
post_features, posts_df = load_post_features()

user_cols = list(user_features.columns)
post_cols = list(post_features.columns)
time_cols = ["hour", "weekday", "is_weekend"]

try:
    FEATURE_COLUMNS = list(model.feature_names_)
    if not FEATURE_COLUMNS:
        raise ValueError("Empty feature_names_")
except Exception:
    
    time_cols = ["hour", "weekday", "is_weekend"]
    user_cols = list(user_features.columns)
    post_cols = list(post_features.columns)

    post_cols = [c for c in post_cols if c != "unique_word_count"]
    FEATURE_COLUMNS = time_cols + user_cols + post_cols




def make_features_user(user_id: int, time: datetime) -> pd.DataFrame:
    """
    Собираем датафрейм для отработки модели
    """

    if user_id not in user_features.index:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_row = user_features.loc[user_id]

    features = post_features.copy()

    for col in user_features.columns:
        features[col] = user_row[col]

    features["hour"] = time.hour
    features["weekday"] = time.weekday()
    features["is_weekend"] = int(features["weekday"].iloc[0] >= 5)

    X = features[FEATURE_COLUMNS].fillna(0)

    X.index.name = "post_id"

    return X


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommendation_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    """
    Эндпоинт для получения топ limit рекомендаций постов для пользователя с id
    """

    X = make_features_user(id, time)

    preds = model.predict_proba(X)[:, 1]

    df_pred = pd.DataFrame(
        {
            "post_id": X.index.values,
            "pred": preds
        }
    )

    top_posts = df_pred.sort_values("pred", ascending=False).head(limit)

    recommendations: List[PostGet] = []

    for post_id in top_posts["post_id"].values:
        row = posts_df.loc[post_id]
        recommendations.append(
            PostGet(
                id=post_id,
                text=row["text"],
                topic=row["topic"]
            )
        )

    return recommendations
