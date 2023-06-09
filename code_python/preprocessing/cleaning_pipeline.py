import sys

sys.path.append("../cleaning")
from clean_r_groups import r_main
from clean_min_structures import man_main
from preprocess_min import pre_main


def pipe_main():
    r_main()
    man_main()
    pre_main()


if __name__ == "__main__":
    pipe_main()
