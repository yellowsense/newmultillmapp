from flask import Flask, render_template, request
import os

app = Flask(__name__)

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

a = "sk-wLQRwTr8RS"
b = "Z1ph7bIlovT3B"
c = "lbkFJe49s56n"
d = "KfeWgts8MpMQC"
oak = a + b + c + d

directory = './datatset'

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)

def split_docs(documents, chunk_size=800, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embeddings)

os.environ["OPENAI_API_KEY"] = oak

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)

chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form.get('query')
    ranking_prompt = " go thoroughly through the document you're trained on and rank by rating in descending order and dont show their phone number. Make sure you dont mention that you have omitted their phone number. "
    user_query_with_ranking = user_query + ranking_prompt
    matching_docs = db.similarity_search(user_query_with_ranking)
    answer = chain.run(input_documents=matching_docs, question=user_query_with_ranking)
    processed_answer = process_answer(answer)
    return render_template('result.html', answer=processed_answer)

def process_answer(answer):
    lines = answer.split('\n')
    processed_lines = []
    for line in lines:
        if "Phone Number:" not in line:
            processed_lines.append(line)
    processed_answer = '<br>'.join(processed_lines)
    return processed_answer

if __name__ == '__main__':
    app.run(debug=True)
