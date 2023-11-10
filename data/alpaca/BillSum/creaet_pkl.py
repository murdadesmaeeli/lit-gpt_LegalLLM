import pandas as pd
import pickle
import json
import sys
sys.path.insert(1,'../../..')
from lit_gpt.tokenizer import Tokenizer
files=['train.jsonl','test.jsonl']
output=[]
tk= Tokenizer('../../../checkpoints/stabilityai/stablelm-tuned-alpha-3b')
for file in files:
    output=[]
    data=json.loads(pd.read_json(file,lines=True).to_json(orient='records'))
    for i,item in enumerate(data):
        stepDict={}
        if len(tk.encode(string=item['article']))>1400:
        #if False:
            continue
        else:
            stepDict['instruction']='Summarize the following legislation in a few words:'
            stepDict['input']=item.pop('article')
            stepDict['output']=item.pop('summary')
            output.append(stepDict)
    with open(f'{file.split(".")[0]}.pkl','wb') as f:
        pickle.dump(output,f)

