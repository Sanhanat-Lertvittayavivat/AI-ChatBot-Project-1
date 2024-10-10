import faiss
# from fastapi import FastAPI
# from pydantic import BaseModel
import requests

from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from sentence_transformers import SentenceTransformer, util, InputExample
from sentence_transformers import models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from neo4j import GraphDatabase, basic_auth
import json
import numpy as np
from time import sleep

# Initialize Sentence Transformer model
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Neo4j connection details
URI = "neo4j://localhost"
AUTH = ("neo4j", "Password")  # Change password as per your config

# Ollama API details
url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}
llama_model = "supachai/llama-3-typhoon-v1.5"

def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
    driver.close()

# Cypher queries to retrieve data from Neo4j
cypher_query_greeting = '''
MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
'''

question_query = '''
MATCH (n:question) RETURN n.name as name, n.msg_reply as reply;
'''

# Load greeting and question data from Neo4j
greeting_corpus = []
results_greeting = run_query(cypher_query_greeting)
for record in results_greeting:
    greeting_corpus.append(record['name'])

question_corpus = []
results_question = run_query(question_query)
for record in results_question:
    question_corpus.append(record['name'])

# Combine both greetings and questions into one corpus
combined_corpus = list(set(greeting_corpus + question_corpus))

# Encode the combined corpus into vectors using Sentence Transformer
corpus_vecs = model.encode(combined_corpus, convert_to_numpy=True, normalize_embeddings=True)

# Initialize FAISS index
d = corpus_vecs.shape[1]  # Dimension of vectors
index = faiss.IndexFlatL2(d)
index.add(corpus_vecs)  # Add vectors to FAISS index

def compute_similar_faiss(sentence):
    try:
        ask_vec = model.encode([sentence], convert_to_numpy=True, normalize_embeddings=True)
        D, I = index.search(ask_vec, 1)  # Search for top 1 result
        return D[0][0], I[0][0]
    except Exception as e:
        print("Error during FAISS search:", e)
        return None, None

def neo4j_search(neo_query):
    results = run_query(neo_query)
    for record in results:
        response_msg = record['reply']
    return response_msg

def llama_response(msg):
    print("this is llama")
    payload = {
        "model": llama_model,
        "prompt": msg + " ตอบเป็นภาษาไทย ให้คุณตอบผมในฐานะที่คุณเป็นผู้เชี่ยวชาญด้านนั้นๆที่ผมถาม ตอบสั้นไม่เกิน 30 คำ และไม่ต้องพูดทวนว่าคุณเป็นใครและจะทำอะไร",
        "stream": False
    }

    print("Sending request to LLaMA API with payload:", json.dumps(payload, indent=2))
    print(f"API URL: {url}")

    try:
        # Attempt to send the request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # Log response details
        print("Received response from LLaMA API:")
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)

        # Check if the response is successful
        if response.status_code == 200:
            try:
                # Attempt to parse the response
                res_JSON = json.loads(response.text)
                res_text = res_JSON.get("response", "No response field in API reply.")
                return res_text
            except Exception as e:
                # Handle any parsing errors
                print("Error while parsing JSON response:", e)
                return "ขออภัย ไม่สามารถประมวลผลคำตอบได้"
        else:
            # Handle unsuccessful HTTP responses
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        # Catch any errors during the request
        print("Error while connecting to LLaMA API:", e)
        return "ขออภัย ไม่สามารถเชื่อมต่อกับเซิฟเวอร์ได้"




def compute_response(sentence):
    distance, index = compute_similar_faiss(sentence)
    Match_input = combined_corpus[index]

    if distance > 0.5:
        my_msg = llama_response(sentence)
        my_msg = "คำตอบนี้มาจาก Ollama :\n" + my_msg
    else:
        if Match_input in greeting_corpus:
            My_cypher = f"MATCH (n:Greeting) WHERE n.name = '{Match_input}' RETURN n.msg_reply as reply"
        else:
            My_cypher = f"MATCH (n:question) WHERE n.name = '{Match_input}' RETURN n.msg_reply as reply"
        
        my_msg = neo4j_search(My_cypher)
    
    return my_msg

# Flask app setup
app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        access_token = 'access_token'
        secret = 'secret_token'
        line_bot_api = LineBotApi(access_token)              
        handler = WebhookHandler(secret)                     
        signature = request.headers['X-Line-Signature']      
        handler.handle(body, signature)
        msg = json_data['events'][0]['message']['text']      
        tk = json_data['events'][0]['replyToken']   

        response_msg = compute_response(msg)
        # response_msg = llama_response(msg)

        line_bot_api.reply_message(tk, TextSendMessage(text=response_msg)) 
        print(msg, tk)                      
        print("-"*20)
        print(response_msg)                
    except:
        print(body)                 
    return 'OK'



@app.route("/api", methods=["POST"])
def api_response():
    body = request.get_data(as_text=True)
    json_data = json.loads(body)
    print("Data received:", json_data)
    response_msg = llama_response(json_data["prompt"])
    return response_msg

if __name__ == '__main__':
    app.run(port=5000)
