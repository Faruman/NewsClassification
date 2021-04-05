import pandas as pd
import numpy as np

from transformers import DistilBertTokenizer, BertTokenizer, XLMTokenizer, RobertaTokenizer
from nltk import word_tokenize
import fasttext
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tqdm import tqdm
tqdm.pandas()

class Tokenizer():
    def __init__(self, args: dict, fasttextFile: str, doLower: bool):
        self.fasttextFile = fasttextFile
        self.doLower = doLower
        self.args = args
        self.tokenizer = None

    def fit(self, series: pd.Series):
        if self.args["tokenizer"] == "bert":
            if self.doLower:
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            else:
                tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            def generate_BERT_vectors(s):
                toks = tokenizer(s,  return_attention_mask= True, padding="max_length", truncation= True)
                return (toks["input_ids"], toks["attention_mask"])
            def tokenizer_fun(series):
                return series.progress_apply(generate_BERT_vectors)
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "distilbert":
            if self.doLower:
                # distilbert german uncased should be used, however a pretrained model does not exist
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            else:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            def generate_DistilBERT_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation= True)
                return (toks["input_ids"], toks["attention_mask"])
            def tokenizer_fun(series):
                return series.progress_apply(generate_DistilBERT_vectors)
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "xlnet":
            if self.doLower:
                # XLNET uncased should be used, however a pretrained model does not exist
                tokenizer = XLMTokenizer.from_pretrained('xlnet-base-cased')
            else:
                tokenizer = XLMTokenizer.from_pretrained('xlnet-base-cased')

            def generate_XLM_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation=True)
                return (toks["input_ids"], toks["attention_mask"])
            self.tokenizer = generate_XLM_vectors
            def tokenizer_fun(series):
                return series.progress_apply(generate_XLM_vectors)
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "roberta":
            if self.doLower:
                # roberta uncased should be used, however a pretrained model does not exist
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            else:
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

            def generate_Roberta_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation=True)
                return (toks["input_ids"], toks["attention_mask"])
            def tokenizer_fun(series):
                return series.progress_apply(generate_Roberta_vectors)
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "distilroberta":
            if self.doLower:
                # distilroberta uncased should be used, however a pretrained model does not exist
                tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
            else:
                tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

            def generate_DistilRoberta_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation=True)
                return (toks["input_ids"], toks["attention_mask"])
            def tokenizer_fun(series):
                return series.progress_apply(generate_DistilRoberta_vectors)
            self.tokenizer = tokenizer_fun

        elif "fasttext" in self.args["tokenizer"]:
            embeddingModel = fasttext.load_model(self.fasttextFile)
            def generate_fasttext_vectors(s):
                words = word_tokenize(s)
                words_embed = [embeddingModel.get_word_vector(w) for w in words if w.isalpha()]
                if "mean" in self.args["tokenizer"]:
                    words_embed = np.column_stack(words_embed).mean(axis=1)
                elif "max" in self.args["tokenizer"]:
                    words_embed = np.column_stack(words_embed).max(axis=1)
                else:
                    pass
                return words_embed
            def tokenizer_fun(series):
                if "mean" in self.args["tokenizer"] or "max" in self.args["tokenizer"]:
                    return np.row_stack(pd.Series(series).progress_apply(generate_fasttext_vectors).values)
                else:
                    return pd.Series(series).progress_apply(generate_fasttext_vectors).values
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "bow":
            vectorizer = CountVectorizer(ngram_range= (1, self.args["ngram"]))
            vectorizer.fit(series.values)
            def tokenizer_fun(series):
                return vectorizer.transform(series.values)
            self.tokenizer = tokenizer_fun

        elif self.args["tokenizer"] == "tfidf":
            vectorizer = TfidfVectorizer(ngram_range= (1, self.args["ngram"]))
            vectorizer.fit(series.values)
            def tokenizer_fun(series):
                return vectorizer.transform(series.values)
            self.tokenizer = tokenizer_fun

    def transform(self, series: pd.Series):
        return self.tokenizer(series)

    def fit_transform(self, series: pd.Series):
        self.fit(series)
        return self.transform(series)