# 품사 태깅
import nltk
from nltk import pos_tag
from nltk import word_tokenize

# 태거 다운로드
nltk.download('averaged_perceptron_tagger')
# 샘플 텍스트 데이터
text_data = "Rachel loves drinking coffee in the morningg"

# 사전 훈련된 품사 태킹을 사용
text_tagged = pos_tag(word_tokenize(text_data))
print(text_tagged)
"""
[('Rachel', 'NNP'), ('loves', 'VBZ'), ('drinking', 'VBG'), ('coffee', 'NN'), ('in', 'IN'), ('the', 'DT'), ('morningg', 'NN')]
"""
