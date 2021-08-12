import re

# removes character present in pattern
def remove_stuff(word):
    pattern = r"[.…”“';]*"
    return re.sub(pattern, "", word)

# assumes ch is at the end -> "token ch" eg. (आग। to आग ।)
# ch = "?" or "।"
def end_token(word, ch):
    return " ".join([word.replace(ch, ""), ch])


# change "आग," or "आग|" to "आग ," or "आग |"
# ch = "-" , "," , "!"
def punctuation_token(string: str, ch: str, first_time: bool = False):
    
    # if ch is present in the string
    if ch in string:
        new_tokens = []
        tokens = string.split(ch)
    
        for token in tokens:

            #  handling ch in the token when first_time
            if first_time:
                if "?" in token:
                    new_tokens.append(end_token(token, ch="?"))
                elif "।" in token:
                    new_tokens.append(end_token(token, ch="।"))
                else:
                    new_tokens.append(token)
            
            # if not first time, skip removing "?", "।"
            else:
                    new_tokens.append(token)

            new_tokens.append(ch)

            # removing any extra commas in the end
            string =  " ".join(new_tokens).rstrip(ch)

    # if no ch in string
    else:
        if first_time and "?" in string:
            string = end_token(string, ch="?")
        elif first_time and "।" in string:
            string = end_token(string, ch="।")
    
    if first_time:
        # removing special symbols
        string = remove_stuff(string)
    
    return string.strip()

def preprocess(textline):
    textline_tokens = textline.split()
    string = ""
    for token in textline_tokens:
        first_time = True
        for ch in ["-", ",", "!"]:
            token  = punctuation_token(token, ch=ch, first_time=first_time)
            first_time = False
        string += " " + token
    return string.strip()

# with open("output.txt", "w") as f:
#        print(preprocess("मेरे सीने में नहीं तो तेरे सीने में| सही,hello,"), file=f)
if __name__== "__main__":
    newline_count = 0
    with open("datasets/temp_data.tsv", "r") as fin:
        with open("datasets/preprocessed_data.tsv", "w") as fout:
            poems=0
            for line in fin:
                
                if line=="\n":
                    newline_count+=1
                
                else:
                    if newline_count<=1:
                        fout.write(preprocess(line)+"\n")
                    
                    elif newline_count>2:
                        fout.write("\n"*3)
                        poems+=1
                        fout.write(preprocess(line)+"\n")
                    newline_count=0
    print(f"poems detected: {poems}")