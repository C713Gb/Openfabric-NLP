import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


# Read science documents from a text file
def read_documents_from_file(file_path):
    with open(file_path, 'r') as file:
        documents = file.readlines()
    return [doc.strip() for doc in documents]


# Preprocess the documents
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = word_tokenize(text.lower())
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)


# Function to get the most relevant document for a given query
def get_most_similar_document(query, documents, vectorizer, tfidf_matrix):
    query = preprocess_text(query)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    most_similar_index = similarities.argmax()
    return documents[most_similar_index]


# Sample science-related documents
file_path = os.getcwd() + '/data.txt'

science_documents = read_documents_from_file(file_path)
processed_documents = [preprocess_text(doc) for doc in science_documents]

# Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_documents)


def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        response = get_most_similar_document(text, science_documents, vectorizer, tfidf_matrix)
        output.append(response)

    return SimpleText(dict(text=output))


if __name__ == "__main__":
    print("Science Chatbot: Hi! I'm here to answer your science-related questions. Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Science Chatbot: Goodbye!")
            break

        mock_request = SimpleText(dict(text=[user_input]))
        response = execute(mock_request, None)
        for resp in response.text:
            print(resp)
