import random
import json

#-----------------------------------------------
#this code is only for the fine-tuning approach
#-----------------------------------------------

#randomly split into 8:2 training and validation datasets
with open('cleaned_data.jsonl', 'r') as f1:
    data = [json.loads(line) for line in f1]


data_size = len(data)
random.shuffle(data)


with open('train.jsonl', 'w') as f2:
    for obj in data[:int(data_size*0.85)]:
        json.dump(obj, f2)
        f2.write('\n')


with open('val.jsonl', 'w') as f3:
    for obj in data[int(data_size*0.85):]:
        json.dump(obj, f3)
        f3.write('\n')