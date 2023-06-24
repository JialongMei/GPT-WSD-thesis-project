import xml.etree.ElementTree as ET
import re
import openai
from nltk.corpus import wordnet as wn



openai.api_key = 'replace with openai api key'
sense_keys = []

with open('WSD_Unified_Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt', 'r') as file:
    lines = file.readlines()

# extract info from gold key file for the given line
def extract_line_info(line):
    split_line1 = line.strip().split('.')
    split_line2 = line.strip().split(' ')

    sen_id = '.'.join(split_line1[:2])
    term_id = split_line2[0]
    keyword = re.search(r'\s(.*?)%', line).group(1)
    sense_key= split_line2[1:]

    return sen_id, term_id, keyword, sense_key

#get the meaning list of the word from WordNet
def get_word_meanings(word):
    global sense_keys

    sense_keys.clear()

    synsets = wn.synsets(word)

    if not synsets:
        return "Cannot provide meaning for this word, so do not choose a number and just answer the meaning of this word in this sentence yourself"

    meanings = []
    for i, synset in enumerate(synsets):
        meanings.append(f"{i}. {synset.definition()}")
        sense_keys.append((i, synset.name()))
    return '\n'.join(meanings)


#ask the model with the word in that specific line
def sense_check(line_number):
    tree = ET.parse('WSD_Unified_Evaluation_Datasets/semeval2007/semeval2007.data.xml')
    root = tree.getroot()

    #get info from the selecting line in gold key file
    sen_id, term_id, keyword, sense_key = extract_line_info(lines[line_number])

    sentence_element = root.find('.//sentence[@id="'+ sen_id +'"]')
    full_sentence = ''.join(sentence_element.itertext())
    full_sentence = full_sentence.replace('\n', ' ')
    full_sentence = full_sentence.strip()

    text = full_sentence
    target_word = keyword

    meanings = get_word_meanings(target_word)

    processed_target_word = re.sub('_', ' ', target_word)

    prompt_content = "What is the meaning of word \"" + processed_target_word + "\" in the sentence: \"" + text + "\"\n" + "Options" + "\n" + meanings + "\n" + "Repeat the option you selected(include the number):" + "\n"
    # print(prompt_content)

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_content,
        max_tokens=33,
        temperature=0,
        top_p=0.3,
        frequency_penalty=0,
        presence_penalty=0
    )

    reply = response.choices[0].text.strip()
    # print(reply)
    sense_key_num = re.search(r'\d+', reply)#locate the first number in the reply text

    synset_id = None
    if sense_key_num:
        chosen_number = int(sense_key_num.group())
        if chosen_number < len(sense_keys): #prevent unexpected behavior from the model
            synset_id = sense_keys[chosen_number][1]

    return check_same_meaning(sense_key, synset_id)


def check_same_meaning(sense_key, synset_id):
    for key in sense_key:
        lemma = wn.lemma_from_key(key)
        if lemma:
            synset = lemma.synset()
            if synset.name() == synset_id:
                return True
    return False