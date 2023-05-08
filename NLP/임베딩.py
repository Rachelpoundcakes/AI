from gensim.models import Word2Vec

sentences = [["apple", "banana"], ["orange", "grape"]]
model = Word2Vec(sentences, min_count=1)

# min_count: 매개변수는 단어가 최소한 몇 번 이상 나와야 학습에 포함되는지를 설정

print(model.wv["apple"])   # 임베딩된 벡터 출력
print(model.wv.most_similar("apple"))   # 가장 유사한 단어 출력
