from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
text = "You don't need to search Iris data on Google but to load by Scikit-Learn"
print(tokenizer.tokenize(text))
"""
['You', 'do', "n't", 'need', 'to', 'search', 'Iris', 
'data', 'on', 'Google', 'but', 'to', 'load', 'by', 'Scikit-Learn']
"""
# TreebankWordTokenizer는 하이픈(-)을 붙여서 하나로 출력한다.
text2 = "바람이 솔솔 부는 곳에서는 가만히 앉아만 있어도 낙원 그 자체이다. -자작글"
print(tokenizer.tokenize(text2))
"""
['바람이', '솔솔', '부는', '곳에서는', '가만히', '앉아만', '있어도', '낙원', '그', '자체 
이다.', '-자작글']
"""