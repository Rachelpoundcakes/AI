# Integer Encoding

from nltk.tokenize import sent_tokenize # method to split a document or paragraph into sentences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text = "Mj started this course last October. She got very nervous and passionate at the starting point of the course. She has yet to finish the course, but she has suffered from many happenings after the course started. Two months left. She will overcome all troubles. Every student will overcome their own troubles. Life is a journey to overcome all troubles."

# Tokenization
sentences = sent_tokenize(raw_text)
print(sentences)
"""
['Mj started this course last October.', 'She got very nervous and passionate at the starting point of the course.', 'She has yet to finish the course, but she has suffered from many happenings after the course started.', 'Two months left.', 'She will overcome all troubles.', 'Every student will overcome their own troubles.', 'Life is a journey to overcome all troubles.']
"""

# dictionary 사용하기
vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))

for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence:
        word = word.lower() # 소문자로 변경하여 단어 개수 줄이기
        if word not in stop_words: # 조건1 불용어 제거
            if len(word) > 2: # 조건2 단어 길이가 2를 초과하는 경우에(=3)
                result.append(word) # 단어를 추가한다
                if word not in vocab: # 딕셔너리 단어 빈도수 기록
                    vocab[word] = 0
                vocab[word] += 1
    preprocessed_sentences.append(result)

print(preprocessed_sentences)
"""
[['started', 'course', 'last', 'october'], ['got', 'nervous', 'passionate', 'starting', 'point', 'course'], ['yet', 'finish', 'course', 'suffered', 'many', 'happenings', 'course', 'started'], ['two', 'months', 'left'], ['overcome', 'troubles'], ['every', 'student', 'overcome', 'troubles'], ['life', 'journey', 'overcome', 'troubles']]
"""
print('단어별 빈도수: ', vocab)
"""
단어별 빈도수:  {'started': 2, 'course': 4, 'last': 1, 'october': 1, 'got': 1, 'nervous': 1, 'passionate': 1, 'starting': 1, 'point': 1, 'yet': 1, 'finish': 1, 'suffered': 1, 'many': 1, 'happenings': 1, 'two': 1, 'months': 1, 'left': 1, 'overcome': 3, 'troubles': 3, 'every': 1, 'student': 1, 'life': 1, 'journey': 1}
"""

print(vocab["course"]) # 4

# 고빈도순으로 정렬하기
"""
# lambda 함수란?
일명 무명함수(익명함수)로, 결과부분을 return 키워드 없이 자동으로 return해준다.
lambda 매개변수 : 표현식 <- 형태로 쓴다.

<def 함수를 lambda 함수로 바꿔보기>

def plus(x, y):
    return x + y

print(plus(10, 20))
>>30

print((lambda x, y: x + y)(10, 20)) # 매개변수 x, y를 받아 x + y의 값으로 return한다.
"""

vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True) # x를 받아 리스트 2번째 요소를 key로 정렬한다.
print(vocab_sorted)
"""
[('course', 4), ('overcome', 3), ('troubles', 3), ('started', 2), ('last', 1), ('october', 1), ('got', 1), ('nervous', 1), ('passionate', 1), ('starting', 1), ('point', 1), ('yet', 1), ('finish', 1), ('suffered', 1), ('many', 1), ('happenings', 1), ('two', 1), ('months', 1), ('left', 1), ('every', 1), ('student', 1), ('life', 1), ('journey', 1)]
"""
# 고빈도순으로 인덱스(1, 2, 3...)를 부여한다.
word_to_index = {}
i = 0
for (word, frequency) in vocab_sorted:
    if frequency > 1: # 빈도수 1 이상인 것만
        i = i + 1
        word_to_index[word] = i
print(word_to_index)
