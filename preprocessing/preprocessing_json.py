import re
import json

# removes character present in pattern
def remove_stuff(word):
    pattern = r"[.…”“';()]*"
    return re.sub(pattern, "", word)

# assumes ch is at the end -> "token ch" eg. (आग। to आग ।)
# ch = "?" or "।"
def end_token(word, ch):
    return " ".join([word.replace(ch, ""), ch])
    
def remove_english_token(string):
    english_pattern = r"[A-Za-z]+"
    result = re.sub(pattern=english_pattern, repl="", string=string)
    return result

def combined_poems(string):
    if string.startswith("-"):  
        return True
    else: 
        return False

# change "आग," or "आग|" to "आग ," or "आग |"
# ch = "-" , "," , "!"
def punctuation_token(string: str, ch: str, first_time: bool = False):
    '''
    Args:
        string: string to be processed
        ch: "-" , "," , "!" (one of the following)
        first_time: boolean to tell if its first time for the line to be preprocessed, 
                    so that subsequent preprocessing do not apply processing for "?", "।"

    Returns:
        string
    '''
    
    # remove english tokens only if first time
    if first_time:
        string = remove_english_token(string)

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


# preprocesses one line at a time
def preprocess(textline):
    textline_tokens = textline.split()
    string = ""
    for token in textline_tokens:
        first_time = True
        for ch in ["-", ",", "!"]:
            token  = punctuation_token(token, ch=ch, first_time=first_time)
            first_time = False
        if string is not None:
            string += " " + token
    return string.strip()

if __name__== "__main__":
    with open("scrapper/scrapped.json", "r") as fin:
        data = json.load(fin)

    poem_data = ""
    for poem in data:
        direct = False  # detects poet's name at the last line and skips adding extra whitespaces
        for line in poem["lines"]:
            if combined_poems(line):
                poem_data+="\n\n\n"
                direct = True
                continue
            
            preprocessed = preprocess(line)
            if preprocessed is not None:
                poem_data+="\n"+preprocessed
                direct = False
        
        if not direct:
            poem_data+="\n\n\n"

    with open("datasets/preprocessed_data.tsv", "w") as fout:
        fout.write(poem_data)