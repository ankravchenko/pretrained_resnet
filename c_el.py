import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

s_el="elephant"
cat_list=[]
elephant_list=[]

f=open('elephant_cat_offsets.txt','r')
for x in f.readlines():
	#print(x)
	#print(x[1:])
	sns=wn.synset_from_pos_and_offset('n',int(x[1:]))
	print(sns)
	if s_el in str(sns):
		elephant_list.append(x)
	else:
		cat_list.append(x)

print("elephant synsets:")
for x in elephant_list:
	sns=wn.synset_from_pos_and_offset('n',int(x[1:]))
	print(x, ": ", str(sns))

print("cat synsets:")
for x in cat_list:
	sns=wn.synset_from_pos_and_offset('n',int(x[1:]))
	print(x, ": ", str(sns))



