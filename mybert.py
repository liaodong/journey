
import numpy as np
from transformers import BertTokenizer
from transformers import  TFBertForMaskedLM

def build_vocab():
    vocab = ['PAD','unused0','unused1','UNK','SOS','SEP','EOS']
    with open('data/raw.txt','r', encoding='utf-8') as f:
        str_temp = f.read()
        if '\r\n' in str_temp:
            str_temp = str_temp.replace('\r\n','')
        elif '\n' in str_temp:
            str_temp = str_temp.replace('\n','')
        for data in str_temp:
            if data not in vocab:
                vocab.append(data)
            else:
                pass
    with open('data/vocab.txt','w') as wf:
        wf.write('\n'.join(vocab))
    pass


def build_tokenizer():
    from tokenizers import Tokenizer, models, trainers

    tokenizer = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(vocab_size=100, min_frequency=2)
    tokenizer.train([
        'data/raw.txt'
    ], trainer=trainer)

    tokenizer.save('model/tokenizer.json', pretty=True)


def preprocess_data(tokenizer):
    with open(r'data/raw.txt', 'r', encoding='utf-8') as f:
        cc = f.read()
    # cc = ['[CLS] 中国的首都是哪里？ [SEP] 北京是 [MASK] 国的首都。 [SEP]']
    train_data = [tokenizer.encode(i, add_special_tokens=False) for i in cc]

    return train_data
    pass


def tokenize():
    pass


def train():
    pass


def predict(model, tokenizer, content):
    cc = tokenizer.encode(content, return_tensors='tf', add_special_tokens=False)
    outputs = model.generate(cc)
    for output in outputs:
        Y = tokenizer.decode(output, skip_special_tokens=True)
        print(Y)
    pass


def load_model():

    model = TFBertForMaskedLM.from_pretrained('clue/roberta_chinese_3L312_clue_tiny', from_pt=True)

    # model = load_trained_model_from_checkpoint(
    #     'fine_model/bert_config.json',
    #     'fine_model/bert_model.ckpt',
    #     training=True,
    #     trainable=True,
    #     seq_len=50
    # )
    # model = BertModel.from_pretrained('fine_model/bert')
    # try:
    #     model = TFGPT2LMHeadModel.from_pretrained(r'model/')
    # except:
    #     config = GPT2Config()
    #     model = TFGPT2LMHeadModel(config=config)

    # tokenizer = BertTokenizer.from_pretrained('fine_model/bert_config.json')
    tokenizer = BertTokenizer.from_pretrained('clue/roberta_chinese_3L312_clue_tiny')

    # tokenizer = Tokenizer.from_file(r'model/tokenizer.json')
    return model, tokenizer

'''
遮蔽语言模型 Masked Language Model
TFBertForMaskedLM
'''

if __name__ == '__main__':
    model, tokenizer = load_model()

    cc = ['[CLS] 中国的首都是哪里？ [SEP] 北京是 [MASK] 国的首都。 [SEP]']
    # dataset = [tokenizer.encode(i, add_special_tokens=False) for i in cc]
    dataset = tokenizer.encode(cc[0], add_special_tokens=False, return_tensors='tf')
    outputs = model.predict(dataset)

    prediction_scores = outputs[0]
    sample = prediction_scores[0]
    pred = np.argmax(sample, axis=1)
    print(pred)
    print(tokenizer.convert_ids_to_tokens(pred))
    print(tokenizer.convert_ids_to_tokens(pred)[14])

