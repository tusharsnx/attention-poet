import shutil

# creating backup first
with open("datasets/data.tsv", "r") as original:
    with open("datasets/backup.tsv", "w") as backup:
        shutil.copyfileobj(original, backup)

# now combining preprocessed_data.tsv
with open("datasets/data.tsv", "a") as original:
    with open("datasets/preprocessed_data.tsv", "r") as new:
        original.write("\n"*3)
        for line in new:
            original.write(line)
