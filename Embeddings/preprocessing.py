import pickle


''' Preprocessing will create strings like "Tar Con Neg Neg Neg...."
This way we can save arbitary number of negative samples.
extra negative samples can be clipped by the user when tokenization using max_seq_len arg.
'''

window = 2

with open("datasets/data.tsv", "r") as fin:
    # storing all inputs in outputs[0] and all targets in outputs[1]
    outputs = [[], []]      
    for ind, line in enumerate(fin):
        word_list = line.split()
        if ind%1000==0:
            print(f"done till linked {ind}...")
        # skip first line
        if ind==0:
            continue
        for i in range(len(word_list)):
            neg_samples = set()     # to remove multiple occurrence of same negative sample word 
            for j in range(len(word_list)):
                if i==j:    # same word
                    continue
                if abs(i-j)<=window:    # context word
                    outputs[0].append([word_list[i], word_list[j]])
                    outputs[1].append(1)
                else:
                    if word_list[j] not in neg_samples:
                        outputs[0].append([word_list[i], word_list[j]])     
                        outputs[1].append(0)
                        
with open("embeddings/embedding_data.pkl", "wb") as fout:
    pickle.dump(outputs, file=fout)
