import logging

def build_model(args):
    if args.token_list is not None:
        with open(args.token_list) as f:
            token_list = [line.rstrip() for line in f]
            args.token_list = list(token_list)
            vocab_size = len(token_list)
            logging.info(f"Vocabulary size: {vocab_size}")




