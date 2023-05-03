# Open-source Korean Text Processor
from konlpy.tag import Okt

okt = Okt()

def build_bow(doc):
    doc = doc.replace('.', '')
    tokenized_doc = okt.morphs(doc)

    word_to_index = {}
    bow = []

    for word in tokenized_dox:
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)

            # BoW에 기본값 1을 넣는다.
            bow.insert(len(word_to_index) -1, 1)
        else:
            # 재등장하는 단어의 인덱스
            index - word_to_index.get(word)
            # 1을 더한다.
            bow[index] = bow[index] + 1

    return word_to_index, bow

doc1 = "저는 인공지능 분야에 취업할 계획입니다. 인공지능은 재미있습니다. 인공지능 분야에서도 특히 자연어 처리에 관심이 있습니다."

vocab, bow = build_bow(doc1)
print('단어: ', vocab)
print('Bag of Words Vector: ', bow)