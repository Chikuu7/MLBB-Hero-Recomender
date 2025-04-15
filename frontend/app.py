import streamlit as st
import requests
from PIL import Image
from io import BytesIO

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="MLBB Recommender", layout="wide")
st.title("🔥 MLBB Hero Recommendation System")

st.sidebar.header("Choose Feature")
feature = st.sidebar.radio("Select an option", ["Lane-based Recommendation", "Compare Heroes", "Pick Rate Chart", "Role Distribution", "Stats Heatmap", "Recommend Similar Heroes"])

if feature == "Lane-based Recommendation":
    st.subheader("🛡️ Lane-based Hero Recommendation")
    lane = st.selectbox("Select a lane", ["gold", "mid", "roam", "jungle", "exp"])
    if st.button("Get Recommendations"):
        res = requests.get(f"{BASE_URL}/recommend_by_lane", params={"lane": lane})
        data = res.json()
        st.table(data)

elif feature == "Compare Heroes":
    st.subheader("⚔️ Compare Two or More Heroes")
    heroes = st.text_input("Enter hero names separated by comma (e.g. martis, lesley, tigreal)")
    if st.button("Compare"):
        hero_list = [h.strip() for h in heroes.split(",")]
        res = requests.post(f"{BASE_URL}/compare_heroes", json={"heroes": hero_list})
        if res.status_code == 200:
            image = Image.open(BytesIO(res.content))
            st.image(image, caption="Hero Stats Comparison", use_column_width=True)
        else:
            st.error("One or more hero names not found.")

elif feature == "Pick Rate Chart":
    st.subheader("📊 Top 10 Heroes by Pick Rate")
    res = requests.get(f"{BASE_URL}/pickrate_chart")
    if res.status_code == 200:
        image = Image.open(BytesIO(res.content))
        st.image(image, caption="Top Picked Heroes", use_column_width=True)

elif feature == "Role Distribution":
    st.subheader("📈 Hero Role Distribution")
    res = requests.get(f"{BASE_URL}/role_distribution")
    if res.status_code == 200:
        image = Image.open(BytesIO(res.content))
        st.image(image, caption="Hero Role Pie Chart", use_column_width=True)

elif feature == "Stats Heatmap":
    st.subheader("📉 Correlation Heatmap of Hero Stats")
    res = requests.get(f"{BASE_URL}/heatmap_stats")
    if res.status_code == 200:
        image = Image.open(BytesIO(res.content))
        st.image(image, caption="Heatmap", use_column_width=True)

elif feature == "Recommend Similar Heroes":
        hero = st.text_input("Enter Hero Name (e.g., Martis):")
        if st.button("Get Recommendations"):
            res = requests.get(f"{BASE_URL}/recommend", params={"hero_name": hero})
            if res.status_code == 200:
                data = res.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.success("Recommended Heroes:")
                    st.table(data)
            else:
                st.error("API Error.")
