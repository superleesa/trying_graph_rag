import json
import re
import ollama


def generate_answer_ollama(query, context_text):
    # Use Ollama to generate the answer based on query and context
    input_text = f"Question: {query} Context: {context_text}"
    response = ollama.generate(model="gemma2:2b", prompt=input_text)  # Adjust model name as necessary
    return response['response']  # Assuming response contains the generated answer in 'response'

# Define the relevance function to identify supporting facts
def model_decides_sentence_is_relevant(sentence, answer):
    # Simple keyword matching for relevance (adjust if necessary)
    sentence_words = set(re.findall(r'\w+', sentence.lower()))
    answer_words = set(re.findall(r'\w+', answer.lower()))
    return len(sentence_words & answer_words) > 0

# Load test data
with open('/Users/sho/Monash/FIT5047/project/chunking_strategies/sampled_gold_data.json', 'r') as f:
    test_data = json.load(f)

# Initialize predictions dictionary
predictions = {"answer": {}, "sp": {}}

# Process each item in test data
for item in test_data:
    question_id = item["_id"]
    question_text = item["question"]
    context = item["context"]

    # Concatenate all context sentences for model input
    context_text = " ".join(" ".join(sentences) for _, sentences in context)

    # Generate answer based on question and context
    answer = generate_answer_ollama(question_text, context_text)

    # Identify supporting facts
    supporting_facts = []
    for title, sentences in context:
        for sent_id, sentence in enumerate(sentences):
            if model_decides_sentence_is_relevant(sentence, answer):
                supporting_facts.append([title, sent_id])

    # Store predictions in the required format
    predictions["answer"][question_id] = answer
    predictions["sp"][question_id] = supporting_facts

# Save predictions to file
with open("dev_distractor_pred.json", "w") as f:
    json.dump(predictions, f, indent=4)

print("Predictions saved to dev_distractor_pred.json")