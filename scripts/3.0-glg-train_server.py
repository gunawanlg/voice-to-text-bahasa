import argparse


def train(args):
    # from IPython.core.display import display, HTML
    from gurih.data.data_generator import DataGenerator
    from gurih.models.model import BaselineASRModel
    from gurih.models.model_utils import CharMap

    # Training parameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    # Preprocess parameters
    MAX_SEQ_LENGTH = args.max_seq_length
    MAX_LABEL_LENGTH = args.max_label_length
    MFCCS = args.mfccs

    # Directories
    train_dir = args.train_dir
    valid_dir = args.valid_dir

    # START TRAINING #
    CHAR_TO_IDX_MAP = CharMap.CHAR_TO_IDX_MAP

    BaselineASR = BaselineASRModel(input_shape=(MAX_SEQ_LENGTH, MFCCS), vocab_len=len(CharMap()))
    BaselineASR.dir_path = ''
    BaselineASR.doc_path = ''
    BaselineASR.compile()

    CTC_INPUT_LENGTH = BaselineASR.model.get_layer('the_output').output.shape[1]

    train_generator = DataGenerator(input_dir=train_dir,
                                    max_seq_length=MAX_SEQ_LENGTH,
                                    max_label_length=MAX_LABEL_LENGTH,
                                    ctc_input_length=CTC_INPUT_LENGTH,
                                    char_to_idx_map=CHAR_TO_IDX_MAP,
                                    batch_size=BATCH_SIZE)

    validation_generator = DataGenerator(input_dir=valid_dir,
                                         max_seq_length=MAX_SEQ_LENGTH,
                                         max_label_length=MAX_LABEL_LENGTH,
                                         ctc_input_length=CTC_INPUT_LENGTH,
                                         char_to_idx_map=CHAR_TO_IDX_MAP,
                                         batch_size=BATCH_SIZE)

    BaselineASR._callbacks(min_delta=1e-4, patience=5)

    BaselineASR.fit_generator(train_generator=train_generator,
                              validation_generator=validation_generator,
                              epochs=EPOCHS)
    # END TRAINING #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train_server',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of batch size. print ')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')

    # Preprocess parameters
    parser.add_argument('--max_seq_length', type=int, default=2500,
                        help='Maximum input sequence length.')
    parser.add_argument('--max_label_length', type=int, default=100,
                        help='Maximum transcript label length.')
    parser.add_argument('--mfccs', type=int, default=39,
                        help='Number of MFCC features.')

    # Directory
    parser.add_argument('--train_dir', type=str, default="train/",
                        help='Training data directory input.')
    parser.add_argument('--valid_dir', type=str, default="valid/",
                        help='Validation data directory input.')
    parser.add_argument('--dir_path', type=str, default="",
                        help='Directory to store model checkpoint.')
    parser.add_argument('--doc_path', type=str, default="",
                        help='Directory to store model plots.')

    args = parser.parse_args()

    train(args)
