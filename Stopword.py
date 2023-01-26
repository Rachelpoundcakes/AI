import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
print(stopwords.words('english')[:5])
# ['i', 'me', 'my', 'myself', 'we']

nltk.download('punkt')
from nltk.tokenize import word_tokenize

input_sentence = "We have studied hard for the exam since last October."
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(input_sentence)
result = []
for w in word_tokens:
    if w not in stop_words:
        result.append(w)
print(word_tokens)
print(result)
"""
['We', 'have', 'studied', 'hard', 'for', 'the', 'exam', 'since', 'last', 'October', '.']
['We', 'studied', 'hard', 'exam', 'since', 'last', 'October', '.']
"""