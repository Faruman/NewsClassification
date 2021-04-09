import pandas as pd
import numpy as np

from transformers import DistilBertTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer
from nltk import word_tokenize
import fasttext
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tqdm import tqdm
tqdm.pandas()

class Tokenizer():
    def __init__(self, args: dict, fasttextFile: str, doLower: bool, max_length= 512):
        self.fasttextFile = fasttextFile
        self.doLower = doLower
        self.args = args
        self.tokenizer = None
        self.max_length = max_length

    def fit(self, series: pd.Series):
        if self.args["tokenizer"] == "bert":
            if self.doLower:
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            else:
                tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            def generate_BERT_vectors(s):
                toks = tokenizer(s,  return_attention_mask= True, padding="max_length", truncation= True, max_length= self.max_length)
                return (toks["input_ids"], toks["attention_mask"])
            def tokenizer_fun(series):
                return pd.Series(series).progress_apply(generate_BERT_vectors).values
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "distilbert":
            if self.doLower:
                # distilbert german uncased should be used, however a pretrained model does not exist
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            else:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            def generate_DistilBERT_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation= True, max_length= self.max_length)
                return (toks["input_ids"], toks["attention_mask"])
            def tokenizer_fun(series):
                return pd.Series(series).progress_apply(generate_DistilBERT_vectors).values
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "xlnet":
            if self.doLower:
                # XLNET uncased should be used, however a pretrained model does not exist
                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            else:
                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

            def generate_XLM_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding=True, truncation=True, max_length= self.max_length)
                return (toks["input_ids"], toks["attention_mask"])
            self.tokenizer = generate_XLM_vectors
            def tokenizer_fun(series):
                return pd.Series(series).progress_apply(generate_XLM_vectors).values
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "roberta":
            if self.doLower:
                # roberta uncased should be used, however a pretrained model does not exist
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            else:
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

            def generate_Roberta_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation=True, max_length= self.max_length)
                return (toks["input_ids"], toks["attention_mask"])
            def tokenizer_fun(series):
                return pd.Series(series).progress_apply(generate_Roberta_vectors).values
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "distilroberta":
            if self.doLower:
                # distilroberta uncased should be used, however a pretrained model does not exist
                tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
            else:
                tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

            def generate_DistilRoberta_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation=True, max_length= self.max_length)
                return (toks["input_ids"], toks["attention_mask"])
            def tokenizer_fun(series):
                return pd.Series(series).progress_apply(generate_DistilRoberta_vectors).values
            self.tokenizer = tokenizer_fun

        elif "fasttext" in self.args["tokenizer"]:
            embeddingModel = fasttext.load_model(self.fasttextFile)
            def generate_fasttext_vectors(s):
                words = word_tokenize(s)
                if "mean" in self.args["tokenizer"]:
                    words_embed = [embeddingModel.get_word_vector(w) for w in words if w.isalpha()]
                    words_embed = np.column_stack(words_embed).mean(axis=1)
                elif "max" in self.args["tokenizer"]:
                    words_embed = [embeddingModel.get_word_vector(w) for w in words if w.isalpha()]
                    words_embed = np.column_stack(words_embed).max(axis=1)
                else:
                    words = words[:self.max_length]
                    words_embed = [embeddingModel.get_word_vector(w) for w in words if w.isalpha()]
                return words_embed
            def tokenizer_fun(series):
                if "mean" in self.args["tokenizer"] or "max" in self.args["tokenizer"]:
                    return np.row_stack(pd.Series(series).progress_apply(generate_fasttext_vectors).values)
                else:
                    return pd.Series(series).progress_apply(generate_fasttext_vectors).values
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "bow":
            vectorizer = CountVectorizer(ngram_range= (1, self.args["ngram"]), lowercase= self.doLower)
            vectorizer.fit(series)
            def tokenizer_fun(series):
                return vectorizer.transform(series)
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "tfidf":
            vectorizer = TfidfVectorizer(ngram_range= (1, self.args["ngram"]), lowercase= self.doLower)
            vectorizer.fit(series)
            def tokenizer_fun(series):
                return vectorizer.transform(series)
            self.tokenizer = tokenizer_fun

    def transform(self, series):
        return self.tokenizer(pd.Series(series))

    def fit_transform(self, series):
        self.fit(pd.Series(series))
        return self.transform(series)