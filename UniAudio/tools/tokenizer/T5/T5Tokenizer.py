from transformers import T5Tokenizer, T5EncoderModel
import sys
from tools.tokenizer.abs_tokenizer import AbsTokenizer
class FrozenT5Embedder(AbsTokenizer):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-base", device="cpu", max_length=100, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   
        self.feature_dim = 768
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        # print('text ', text.shape)
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # print('batch_encoding ', batch_encoding)
        # assert 1==2
        att_mask = batch_encoding['attention_mask']
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        att_sum = att_mask.sum(1)
        # print(att_mask, att_sum)
        # assert 1==2
        z = outputs.last_hidden_state
        z = z[:,:att_sum,:]
        # print('z ', z.shape)
        z = z.squeeze(0) # return 77, 768
        return z
    
    def tokenize(self, x, task=None, cache=None):
        if isinstance(x, str):
            # the input is the raw text
            return self(x)
        elif x.shape[1] == self.feature_dim:
            # the input is the features
            return x
        else:
            raise NotImplementedError
    
    @property
    def is_discrete(self):
        return False
    
    @property
    def codebook_length(self):
        return 1
    
    def find_length(self, x):
        batch_encoding = self.tokenizer(x, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        #print('batch_encoding ', batch_encoding)
        att_mask = batch_encoding['attention_mask']
        att_sum = att_mask.sum(1)
        return att_sum.item()

if __name__ == '__main__':
    tokenizer = FrozenT5Embedder()
    text = 'it is the time' 
    #codec = tokenizer.tokenize(text)
    ln = tokenizer.find_length(text)
    print(ln)