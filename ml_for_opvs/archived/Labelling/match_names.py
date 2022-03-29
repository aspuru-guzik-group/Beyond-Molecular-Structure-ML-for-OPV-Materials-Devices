import pkg_resources
import pandas as pd

CSV_PATH = pkg_resources.resource_filename(
    "r_group", "Labelling/duplicate_check_acceptor.csv"
)

NEW_CSV_PATH = pkg_resources.resource_filename("r_group", "Labelling/matches.csv")

df = pd.read_csv(CSV_PATH)


index = 0
counter = 0
for mol in df["Match"]:
    for name in df["Named"]:
        if mol == name:
            df["Matched"][index] = name
            counter += 1
    index += 1

# print("Matched: ", counter)
index = 0
for mol in df["Match"]:
    if mol not in df["Named"].tolist():
        df["nomatch1"][index] = mol
    index += 1

index = 0
for mol in df["Named"]:
    if mol not in df["Match"].tolist():
        df["nomatch2"][index] = mol
    index += 1

print(df.count())

df.to_csv(NEW_CSV_PATH)

