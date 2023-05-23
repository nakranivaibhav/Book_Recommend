
import gradio as gr
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# load resources
df = pd.read_csv('final_2.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('/Users/vaibhav/Book_Recommend/model')
index = faiss.read_index('index_file')

# map each document ID to its index in the original dataframe
id_mapping = np.array(range(0, len(df)))

def search(query: str, k=3):
    query_vector = model.encode([query], convert_to_tensor=True)
    query_vector = query_vector / query_vector.norm()  # normalize for cosine similarity
    query_vector_np = query_vector.cpu().numpy()
    _, I = index.search(query_vector_np, k)
    return df.iloc[id_mapping[I[0]].tolist()]

 # return the results as a dictionary
def query(query:str):
    results = search(query)
    
    return results[['Title', 'Authors', 'BuyLink']].to_dict('records')

demo = gr.Interface(fn=query, inputs="text", outputs=gr.outputs.JSON(),
                     title='Suggest a Book',
                     description='No Titles! No Authors! Pour your heart out ♥')

demo.launch()

