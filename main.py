from fastapi import FastAPI, Query, Response
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import seaborn as sns
import io
import uvicorn

# Load the dataset
df = pd.read_csv("mlbb_heros.csv")

# ----------------------
# Data Preprocessing
# ----------------------
# Strip column names of extra whitespace
df.columns = df.columns.str.strip()

# Handle missing values if any (fill with 0)
df.fillna(0, inplace=True)

# Lowercase all hero names and roles for case-insensitive matching
df['hero_name'] = df['hero_name'].str.strip().str.lower()
df['role'] = df['role'].str.strip().str.lower()

# Preprocess dataset
feature_columns = [
    "defense_overall", "offense_overall", "skill_effect_overall",
    "difficulty_overall", "win_rate", "pick_rate"
]

X = df[feature_columns].fillna(0)
knn = NearestNeighbors(n_neighbors=6, algorithm='auto')
knn.fit(X)

app = FastAPI()

# Enable CORS for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompareRequest(BaseModel):
    heroes: List[str]

@app.get("/recommend")
def recommend_heroes(hero_name: str):
    hero_name = hero_name.strip().lower()
    if hero_name not in df.hero_name.values:
        return {"error": "Hero not found"}

    hero_index = df[df.hero_name == hero_name].index[0]
    distances, indices = knn.kneighbors([X.iloc[hero_index]])

    recommended = df.iloc[indices[0][1:]]  # skip the first (it's the hero itself)
    return recommended[['hero_name', 'role', 'win_rate', 'pick_rate']].to_dict(orient="records")

@app.get("/recommend_by_lane")
def recommend_by_lane(lane: str = Query(..., description="Choose from gold, mid, roam, jungle, exp")):
    lane = lane.strip().lower()
    role_map = {
        "gold": ["marksman"],
        "mid": ["mage", "support"],
        "roam": ["tank", "support"],
        "jungle": ["assassin", "fighter"],
        "exp": ["fighter", "tank"]
    }
    roles = role_map.get(lane)
    if not roles:
        return {"error": "Invalid lane name"}

    filtered = df[df['role'].isin(roles)]
    top = filtered.sort_values(by="win_rate", ascending=False).head(5)
    return top[['hero_name', 'role', 'win_rate', 'pick_rate']].to_dict(orient="records")

@app.get("/pickrate_chart")
def pickrate_chart():
    top = df.sort_values(by="pick_rate", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top['pick_rate'], y=top['hero_name'], palette="viridis")
    plt.xlabel("Pick Rate")
    plt.ylabel("Hero Name")
    plt.title("Top 10 Heroes by Pick Rate")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
@app.post("/compare_heroes")
def compare_heroes(data: CompareRequest):
    hero_names = [name.strip().lower() for name in data.heroes]
    comparison = df[df.hero_name.isin(hero_names)][[
        'hero_name', 'win_rate', 'pick_rate', 'offense_overall',
        'defense_overall', 'skill_effect_overall', 'difficulty_overall'
    ]]

    melted = comparison.melt(id_vars='hero_name', var_name='Stat', value_name='Value')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='hero_name', y='Value', hue='Stat')
    plt.title("Hero Stats Comparison")
    plt.xlabel("Hero Name")
    plt.ylabel("Value")
    plt.legend(title="Stat", bbox_to_anchor=(1.05, 1), loc='upper left')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/role_distribution")
def role_distribution():
    role_counts = df['role'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(role_counts, labels=role_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Hero Role Distribution")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/heatmap_stats")
def heatmap_stats():
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[feature_columns].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap of Hero Stats")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

