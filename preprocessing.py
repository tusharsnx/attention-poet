import re

def vertical_bar_token(word):
    if "|" in word:
        return " ".join([word.replace("|", ""), "|"])


# change "आग," or "आग|" to "आग ," or "आग |"
def punctuation_token(string: str):
    
    # handling "," in the token
    if "," in string:
        new_tokens=[]
        tokens = string.split(",")
    
        for token in tokens:

            #  words like चाहिए।
            #  handling "|" in the token
            if "|" in token:
                new_tokens.append(vertical_bar_token(token).replace(".", ""))
                print(new_tokens)

            else:
                new_tokens.append(token.replace(".", ""))
                print(new_tokens)

            new_tokens.append(",")
            # print(new_tokens)
        # removing any extra commas in the end
        return " ".join(new_tokens).rstrip(",").strip()

    else:
        if "|" in string:
            string = vertical_bar_token(string.replace(".", ""))
        return string.strip()

def preprocess(textline):
    textline_tokens = textline.split()
    string = ""
    for token in textline_tokens:
        if "," in token or "|" in token or "." in token:
            string += " " + punctuation_token(token)
        else:
            string += " " + token
    return string.strip()

# with open("output.txt", "w") as f:
#     print(preprocess("मेरे सीने में नहीं तो तेरे सीने में| सही,hello,"), file=f)

with open("temp_data.tsv", "r") as fin:
    with open("preprocess_data.tsv", "w") as fout:
        for line in fin:
            print(line)
            fout.write(preprocess(line)+"\n")