import json

with open("seeds.json", "r") as f:
    SEEDS: list[int] = json.load(f)
