import os
import numpy as np
no_of_sequences=15
sequence_length=30
startfolder=30
actions=['hello','rock','fuck','thumsup','Love','you','peace','koreanlove','super','stop']
data_path=os.path.join('mpdata4')
print(actions)
for action in actions:
    try:
        os.makedirs(os.path.join(data_path,action))
        for i in range(sequence_length):
            os.makedirs(os.path.join(data_path,action,str(i)))
    except:
        pass