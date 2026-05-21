import streamlit as st
import pandas as pd
import warnings
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')
ps = PorterStemmer()

# ========== DATA PROCESSING ==========
def stem_text(text):
    return " ".join([ps.stem(word) for word in text.split()])

def load_data(path):
    data = pd.read_csv(path)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    for col in ['Description', 'Reason']:
        data[col] = data[col].apply(lambda x: x.split())
        data[col] = data[col].apply(lambda x: [i.replace(" ", "") for i in x])

    data['tags'] = data['Description'] + data['Reason']
    data['tags'] = data['tags'].apply(lambda x: " ".join(x).lower())
    data['tags'] = data['tags'].apply(stem_text)

    return data[['Drug_Name', 'tags']]

def build_model(data):
    cv = CountVectorizer(stop_words='english', max_features=5000)
    vectors = cv.fit_transform(data['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity, data

def recommend_drugs(medicine_name, data, similarity_matrix):
    if medicine_name not in data['Drug_Name'].values:
        return ["❌ Medicine not found in dataset."]
    idx = data[data['Drug_Name'] == medicine_name].index[0]
    distances = similarity_matrix[idx]
    similar = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    return [data.iloc[i[0]]['Drug_Name'] for i in similar]

# ========== LOAD DATA ==========
csv_path = "medicine.csv"
data = load_data(csv_path)
similarity, new_data = build_model(data)

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="💊 Medicine Recommender", layout="centered")

st.title("💊 Medicine Recommender System")
st.markdown("Get alternative medicine suggestions based on description and reason.")

selected_medicine = st.selectbox("Select a medicine:", sorted(new_data['Drug_Name'].unique().tolist()))

if st.button("Recommend"):
    recommendations = recommend_drugs(selected_medicine, new_data, similarity)
    st.subheader("🔍 Recommended Alternatives:")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
