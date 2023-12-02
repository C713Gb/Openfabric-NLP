import os
import warnings
import tensorflow as tf
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

from transformers import pipeline, AutoTokenizer, TFAutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = TFAutoModelForQuestionAnswering.from_pretrained('allenai/scibert_scivocab_uncased', from_pt=True)

# Initialize the model once (outside the function to avoid reloading it on each call)
science_qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2",
                               tokenizer="deepset/roberta-base-squad2")


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        print("text ==> ", text)
        response = process_science_query(text)  # process_science_query is a placeholder for your NLP logic
        output.append(response)

    formatted_output = '\n'.join(output)
    response_text = SimpleText(dict(text=formatted_output))

    # Print the content of the SimpleText object for debugging
    print("Response content:", response_text.text)

    return response_text


def process_science_query(query):
    inputs = tokenizer.encode_plus(query, add_special_tokens=True, return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()[0]

    outputs = model(inputs)
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    answer_end = (tf.argmax(answer_end_scores, axis=1) + 1).numpy()[0]

    # Debugging: Print answer start and end positions
    print("Answer start:", answer_start)
    print("Answer end:", answer_end)

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    # Debugging: Print the extracted answer
    print("Extracted answer:", answer)

    return answer
