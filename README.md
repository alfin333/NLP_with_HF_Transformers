<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Muhammad Alfin

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("Not worth the price at all.")
```

Result : 

```
[{'label': 'NEGATIVE', 'score': 0.9998026490211487}]
```

Analysis on example 1 : 

The sentiment analysis classifier accurately detects the positive tone in the given sentence. It shows a high confidence score, indicating that the model is reliable for straightforward emotional expressions, such as enthusiasm or joy, in English-language input.


### 2. Example 2 - Topic Classification

```
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "In recent years, artificial intelligence has made significant strides in the healthcare industry. From diagnostic algorithms to robotic surgery, AI is transforming the way medical professionals approach patient care. Hospitals are now using machine learning models to predict patient outcomes, reduce errors, and improve efficiency in administrative tasks. Despite concerns over data privacy and ethical considerations, the integration of AI into healthcare systems continues to expand at a rapid pace.",
    candidate_labels=["Technology", "Health", "Education", "Politics", "Business"],
)
```

Result : 

```
Device set to use cuda:0
{'sequence': 'In recent years, artificial intelligence has made significant strides in the healthcare industry. From diagnostic algorithms to robotic surgery, AI is transforming the way medical professionals approach patient care. Hospitals are now using machine learning models to predict patient outcomes, reduce errors, and improve efficiency in administrative tasks. Despite concerns over data privacy and ethical considerations, the integration of AI into healthcare systems continues to expand at a rapid pace.',
 'labels': ['Technology', 'Health', 'Business', 'Politics', 'Education'],
 'scores': [0.5502668023109436, 0.3221483826637268, 0.08161081373691559, 0.027751589193940163, 0.018222281709313393]}
```

Analysis on example 2 : 

The zero-shot classifier correctly identifies "pet" as the most relevant label, with a high confidence score. This shows the model's strong ability to associate descriptive context with predefined categories, even without task-specific fine-tuning or training on the input text.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO :
generator = pipeline("text-generation", model="distilgpt2") # or change to gpt-2
generator(
    "this money can make you",
    max_length=15, # you can change this
    num_return_sequences=1, # and this too
)
```

Result : 

```
[{'generated_text': 'this money can make you want to run a company and it’s hard to do that.\n\n\nI’d like to say that since the only one that is out there is a few people who actually do this. I’d love to see what other people do and what other people do and what other people don’t do.\n(And I have a lot of other people who do it all and I’m not the only one.)'}]
```

Analysis on example 3 : 

The text generation model produces coherent and imaginative continuations of a cooking-themed prompt. It demonstrates creativity and sentence flow, although output content may vary in tone and logic. The results showcase the model's usefulness for generating casual or narrative text.

```
#TODO
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("My girlfriend is so <mask> and charmy", top_k=4)
```

Result : 

```
[{'score': 0.30487656593322754,
  'token': 11962,
  'token_str': ' cute',
  'sequence': 'My girlfriend is so cute and charmy'},
 {'score': 0.18134312331676483,
  'token': 4045,
  'token_str': ' sweet',
  'sequence': 'My girlfriend is so sweet and charmy'},
 {'score': 0.03694342076778412,
  'token': 9869,
  'token_str': ' lovely',
  'sequence': 'My girlfriend is so lovely and charmy'},
 {'score': 0.036360710859298706,
  'token': 15652,
  'token_str': ' adorable',
  'sequence': 'My girlfriend is so adorable and charmy'}]
```

Analysis on example 3.5 : 

The fill-mask pipeline accurately infers masked words based on context. The top result "stole" makes sense, supported by a high confidence score. Other predictions are also contextually appropriate, illustrating the model's nuanced understanding of sentence structure and intent.

### 4. Example 4 - Name Entity Recognition (NER)

```
#TODO:
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Muhammad Alfin, a student at SMA NEGERI 2 KUTA passionate about web development, data science, and machine learning!")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.9992709),
  'word': 'Muhammad Alfin',
  'start': 11,
  'end': 25},
 {'entity_group': 'ORG',
  'score': np.float32(0.948416),
  'word': 'SMA NEGERI 2 KUTA',
  'start': 40,
  'end': 57}]
```

Analysis on example 4 : 

The named entity recognizer successfully identifies personal, organizational, and location entities from the sentence. Grouped outputs are relevant and accurate, with high confidence scores, demonstrating the model’s effectiveness in real-world applications like information extraction or document tagging.

### 5. Example 5 - Question Answering

```
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What natural light appears and circle shaped in the sky during the night?"
context = "natural light appears and circle shaped in the sky during the night is moon"
qa_model(question = question, context = context)
```

Result : 

```
{'score': 0.9971681833267212, 'start': 71, 'end': 75, 'answer': 'moon'}
```

Analysis on example 5 : 

The question-answering model correctly extracts the most relevant phrase "a cat" from the provided context. Its confidence score is decent, and the model showcases strong capabilities in understanding natural questions and matching them with the most likely answer span.

### 6. Example 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
Artificial intelligence (AI) is rapidly transforming various industries across the globe. 
From healthcare to finance, AI technologies are improving efficiency, accuracy, and decision-making processes. 
In healthcare, AI-powered tools assist doctors in diagnosing diseases earlier and recommending personalized treatments. 
The finance sector uses AI for fraud detection and algorithmic trading, reducing risks and maximizing profits. 
However, the rise of AI also brings challenges such as ethical concerns, job displacement, and data privacy issues. 
Governments and organizations worldwide are working to create regulations that ensure AI development benefits society 
while minimizing negative impacts. Despite these challenges, the potential for AI to revolutionize how we live and work 
remains vast and promising, making it a key area of focus for future technological innovation.
"""
)
```

Result : 

```
[{'summary_text': ' Artificial intelligence (AI) is rapidly transforming various industries across the globe . From healthcare to finance, AI technologies are improving efficiency, accuracy, and decision-making processes . The rise of AI also brings challenges such as ethical concerns, job displacement, and data privacy issues . Despite these challenges, the potential for AI to revolutionize how we live and work remains vast .'}]

```

Analysis on example 6 :

The summarization pipeline effectively condenses the core idea of the paragraph into a shorter version. It maintains key concepts like machine learning, pattern recognition, and practical applications, reflecting the model's strength in content compression without major loss of information.

### 7. Example 7 - Translation

```
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-en")
translator_id("aku suka kucing dan anjing")
```

Result : 

```
[{'translation_text': 'I love cats and dogs.'}]

```

Analysis on example 7 :

The translation model delivers an accurate and context-aware French translation of the Indonesian sentence. It handles informal, conversational input smoothly, making it suitable for multilingual communication tasks and cross-language understanding in casual or daily scenarios.

---

## Analysis on this project

This project offers a practical introduction to various NLP tasks using Hugging Face pipelines. Each example is easy to follow and demonstrates real-world use cases. The variety of models shows the flexibility of transformer-based solutions in solving different types of language problems.
