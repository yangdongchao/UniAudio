import torch

class AbsTokenizer(torch.nn.Module):
    """
    This is the virtual tokenizer class.
    Other tokenizers should inherit this class.
    typicially:
        Text -> BPE
        Text -> Phone
        Audio -> Codec
        Image -> Codec
        ...
    """

    @property
    def is_discrete(self):
        """ 
        Return True if the results are discrete token-ids: e.g., BPE / Phone / Codec 
        Return False if the results are continuous embeddings: e.g., RoBERTa embeddings
        """
        raise NotImplementedError

    @property
    def codebook_length(self):
        """
        Return 0 if "self.is_discrete is False",
        otherwise returns the length of codebook.
        e.g., for audio codec that adopts 4 codebooks, each of which is in size of 1024,
          this is 4 * 1024
        This is used to create the shared vocabulary for softmax
        """
        raise NotImplementedError

    def find_length(self, x):
        """
        This method quickly returns the length of the output (usually without tokenization)
        This method is used in batchfying process: measure the whole length of the example
        typically:
            number of BPE / Frames / Codec sequence / Embedding lengths
        """
        raise NotImplementedError

    def tokenize(self, x):
        """ Do tokenization.
            typically, x can be any input type, e.g.,
                text: which is a path of the audio
                text: which is the exact text data for BPE / G2P
                Tensor: the loaded data. e.g., audio 
            Returns 1-D LONG tensor when this is discrete
            Returns 2-D FLOAT tensor when this is continuous: [length, embedding_size]
        """
        raise NotImplementedError

    def tokenize_batch(self, xs, lengths=None):
        """ batch version of tokenization
            Implementation of this method is optional, as it will only be used offline.
 
            warning: you should verify that the results of 'tokenize_batch' and 'tokenize'
            are actually (or roughly) identical (i.g., padding will not effect the results)

            return: list of 'tokenize' results. do NOT make it as a batched Tensor
        """
        raise NotImplementedError

    def detokenize(self, x):
        """ This method recovers the original input based on the 'tokenize' result 
            Implementation of this method is optional, as some tokenization process
            is not recoverable. i.g., hubert
        """
        raise NotImplementedError
