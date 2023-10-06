import argparse
from utils.train_utils import str2bool

def add_tokenizer_arguments(parser):
    # This function adds all hyper-parameters about tokenizers.
    # They are activated by default. TODO: turn them inactivated by default to save memory
    # as we don't need all of them in a single task.

    # audio tokenizer
        # audio tokenizer
    parser.add_argument("--audio-tokenizer", type=str, default="none", choices=["soundstream", 'encodec', "none"],
                        help="choice of audio tokenizer")
    parser.add_argument("--audio-tokenizer-select-every", type=int, default=1, 
                        help="set to n_codebook to select the first layer only")

    # audio prompt tokenier
    parser.add_argument("--audio-prompt-tokenizer", type=str, default="none", choices=["audio_prompt", "none"],
                        help="choice of audio prompt tokenizer")
    parser.add_argument("--audio-prompt-length", type=int, default=3,
                        help="audio prompt length in seconds.")

    # phone tokenier
    parser.add_argument("--phone-tokenizer", type=str, default="none", choices=["g2p", "alignment", "none"],
                        help="choice of phone tokenizer")
    parser.add_argument('--phone-tokenizer-dict', type=str, 
                        default='tools/tokenizer/phone/phone_dict',
                        help="phone dict to use if this is the alignment phone tokenizer")

    # text tokenier
    parser.add_argument("--text-tokenizer", type=str, default="none", choices=["bpe", "none"],
                        help="choice of text tokenizer")

    # semantic tokenier
    parser.add_argument("--semantic-tokenizer", type=str, default="none", choices=["hubert", "none"],
                        help="choice of phone tokenizer")
    parser.add_argument("--semantic-tokenizer-duplicate", type=str2bool, default=True,
                        help="if true, the semantic token will not remove duplications")
    
    parser.add_argument("--FrozenT5Embedder", type=str, default="none", choices=["text_t5", "none"],
                        help="choice of FrozenT5Embedder tokenizer")

    parser.add_argument("--text_clip_tokenizer", type=str, default="none", choices=["text_clip", "none"])

    # SV bool tokenizer
    parser.add_argument("--sv-bool-tokenizer", type=str, default="none", choices=["sv_bool", "none"],
                        help="bool tokenizer for SV task")

    # sing related
    parser.add_argument("--singPhoneTokenizer", type=str, default="none", choices=["sing_phone", "none"],
                        help="choice of singPhoneTokenizer tokenizer")
    
    parser.add_argument("--singMidiTokenizer", type=str, default="none", choices=["sing_midi", "none"],
                        help="choice of singMidiTokenizer tokenizer")
    


    
