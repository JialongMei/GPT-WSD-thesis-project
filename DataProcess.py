import nltk
from nltk.corpus import stopwords
from nltk.corpus import semcor
from nltk.corpus import wordnet
import json

#-----------------------------------------------
#this code is only for the fine-tuning approach
#It covert SemCor data to right format
#-----------------------------------------------

# only need once for downloading
# nltk.download('semcor')
# nltk.download('wordnet')
# nltk.download('stopwords')

sentences = semcor.sents()
taggedSen = semcor.tagged_sents()

sents_Len = len(sentences)
word = ""
sentence = ""
answer = ""

stop_words = set(stopwords.words('english'))

def pos_Converter(tag):
    if tag is None:
        return None
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def tag_fetch(targeted_word, pos_tag):
    synsets = wordnet.synsets(targeted_word, pos=pos_tag)

    if synsets:
        first_synset = synsets[0]
        definition = first_synset.definition()
        return definition
    else:
        return False #'No synsets found for the word', word, 'and POS tag', pos_tag

with open('cleaned_data.jsonl', 'w') as f:
    for sent_num in range(sents_Len):
        chunk_len = len(taggedSen[sent_num])
        for chunk_num in range(chunk_len):
            tree_size = len(taggedSen[0][0].leaves())
            for word_num in range(tree_size):
                word = taggedSen[sent_num][chunk_num].leaves()[word_num]
                if word in stop_words:
                    break
                no_empty = [el for el in sentences[sent_num] if el != '``' and el != "''"]
                sentence = " ".join(no_empty)
                answer = tag_fetch(word, pos_Converter(taggedSen[sent_num][chunk_num].label()))
                if not answer:
                    break
                json.dump({"prompt": "Define " + word + " in the following sentence\n\n" + sentence + " \n\n", "completion": answer}, f)
                f.write('\n')

