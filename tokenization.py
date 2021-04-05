import pandas as pd
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
            self.tokenizer = generate_BERT_vectors

        elif self.args["tokenizer"] == "distilbert":
            if self.doLower:
                # distilbert german uncased should be used, however a pretrained model does not exist
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            else:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            def generate_DistilBERT_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation= True)
                return (toks["input_ids"], toks["attention_mask"])
            self.tokenizer = generate_DistilBERT_vectors

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

        elif self.args["tokenizer"] == "roberta":
            if self.doLower:
                # roberta uncased should be used, however a pretrained model does not exist
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            else:
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

            def generate_Roberta_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation=True)
                return (toks["input_ids"], toks["attention_mask"])
            self.tokenizer = generate_Roberta_vectors

        elif self.args["tokenizer"] == "distilroberta":
            if self.doLower:
                # distilroberta uncased should be used, however a pretrained model does not exist
                tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
            else:
                tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

            def generate_DistilRoberta_vectors(s):
                toks = tokenizer(s, return_attention_mask=True, padding="max_length", truncation=True)
                return (toks["input_ids"], toks["attention_mask"])
            self.tokenizer = generate_DistilRoberta_vectors

        elif "fasttext" in self.args["tokenizer"]:
            embeddingModel = fasttext.load_model(self.fasttextFile)
            def generate_fasttext_vectors(s):
                words = word_tokenize(s)
                words_embed = [embeddingModel.get_word_vector(w) for w in words if w.isalpha()]
                if "average" in self.args["tokenizer"]:
                    words_embed
                elif "max" in self.args["tokenizer"]:
                    words_embed
                else:
                    pass
                return words_embed
            self.tokenizer = generate_fasttext_vectors

        elif self.args["tokenizer"] == "bow":
            vectorizer = CountVectorizer(ngram_range= (1, self.args["ngram"]))
            vectorizer.fit(series)
            self.tokenizer = vectorizer.transform

        elif self.args["tokenizer"] == "tfidf":
            vectorizer = TfidfVectorizer(ngram_range= (1, self.args["ngram"]))
            vectorizer.fit(series)
            self.tokenizer = vectorizer.transform

    def transform(self, series: pd.Series):
        return series.progress_apply(self.tokenizer)

    def fit_transform(self, series: pd.Series):
        self.fit(series)
        return self.transform(series)