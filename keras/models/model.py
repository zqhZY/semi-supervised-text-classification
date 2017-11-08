from cnn import get_cnn
from rnns import get_lstm

def get_model(args):
    """
    get models
    """
    if args.mode == "cnn":
        return get_cnn(args)
    elif args.mode == "lstm":
        return get_lstm(args)
