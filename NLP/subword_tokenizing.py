from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
print(tokenizer.tokenize("어머니 가방에 들어가신다"))
# ['어머니', '가', '##방', '##에', '들어', '##가', '##신', '##다']

