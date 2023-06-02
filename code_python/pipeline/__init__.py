import json

with open("filters.json", "r") as f:
    FILTERS: dict[str, list[str]] = json.load(f)

with open("subsets.json", "r") as f:
    SUBSETS: dict[str, list[str]] = json.load(f)

with open("hyperopt_space.json", "r") as f:
    HYPEROPT_SPACE: dict[str, dict[str, list]] = json.load(f)
