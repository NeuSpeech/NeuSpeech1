# 这个文件用来帮助产生更好的生成设置
# 1, 生成sequence bias

import transformers
from transformers import AutoTokenizer
import yake
import jsonlines


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts


class GetSequenceBias(object):
    def __init__(self,tokenizer_name,jsonl_path,bias=None,extract_type=None):
        language = "en"
        max_ngram_size = 3 #最大关键词语长度
        deduplication_threshold = 0.9 #设置在关键词中是否可以重复单词
        numOfKeywords = 20
        self.jsonl=read_jsonlines(jsonl_path)
        self.sentences=[line['sentence'] for line in self.jsonl]
        self.tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
        self.kw_extractor=yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

        self.bias=self.get_bias_for_sentences(self.sentences,bias,extract_type)
        assert self.bias!={}

    def get_phrases_from_sentence(self,sentence,cannot_be_single_word=False):
        phrases=self.kw_extractor.extract_keywords(sentence)
        if not cannot_be_single_word: # 可以包含单个单词
            phrases=[phrase[0] for phrase in phrases]
        else:  # 不包含单个单词
            phrases=[phrase[0] for phrase in phrases if len(phrase[0].split())!=1]
        return phrases

    def get_phrases_from_sentences(self,sentences,cannot_be_single_word):
        unique_sentences=list(set(sentences))
        phrases=[]
        for i,sentence in enumerate(unique_sentences):
            phrases.extend(self.get_phrases_from_sentence(sentence,cannot_be_single_word))
        return phrases

    def get_tokens_as_tuple(self,word):
        return tuple(self.tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])

    def get_tokens_as_tuple_from_sentences(self,sentences):
        # 把句子列表里面的所有单个word拿出来
        words={word for sentence in sentences for word in sentence.split()}
        word_tokens={self.get_tokens_as_tuple(word) for word in words}
        return word_tokens

    def get_bias_for_tokens(self,tokens,bias):
        return {token: bias for token in tokens}

    def get_bias_for_sentences(self,sentences,bias,extract_type=None):
        if extract_type=='word':  # 只包含word
            tokens=self.get_tokens_as_tuple_from_sentences(sentences)
        elif extract_type=='phrase':  # 只包含phrase
            phrases=self.get_phrases_from_sentences(sentences,cannot_be_single_word=True)
            tokens={self.get_tokens_as_tuple(word) for word in phrases}
        elif extract_type=='phrase_word':  # 包含phrase和word
            phrases=self.get_phrases_from_sentences(sentences,cannot_be_single_word=False)
            tokens={self.get_tokens_as_tuple(word) for word in phrases}
        else:
            raise NotImplementedError
        return {token: bias for token in tokens}

    def get_bias_for_my_sentences(self):
        return self.bias


