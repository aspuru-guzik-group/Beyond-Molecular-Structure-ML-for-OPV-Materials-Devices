from argparse import ArguementParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", action="store", type=str, help="Name of model.")
    parser.add_argument("--model", action="store", type=str, help="Name of feature.")
    parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults to 1.")

    FLAGS = parser.parse_args()

    # TODO ... 
    # run the training
    

