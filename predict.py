#TODO: fix fit transform problem allow to save tokenizer and preprocessor

import json
import wandb
import logging
import os
import sys
import random
import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch import optim

from preprocessing import Preprocessor
from tokenization import Tokenizer

from modeling import Model

from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

def pd_load_multiple_files(path, encoding):
    if path.split(".")[-1] == "csv":
        return(pd.read_csv(path, encoding= encoding))
    elif (path.split(".")[-1] == "xlsx") or (path.split()[-1] == "xls"):
        return(pd.read_excel(path, encoding= encoding))
    elif path.split(".")[-1] == "pkl":
        return(pd.read_pickle(path, encoding= encoding))
    else:
        logging.error("{} datatype not supported.".format(path))
        sys.exit("{} datatype not supported.".format(path))

#created custom to gain optimal benefit from precision recall tradeoff
#decision_dict = {
#        "ambience_neg": 0.34,
#        "ambience_pos": 0.46,
#        "drinks_neg": 0.13,
#        "drinks_pos": 0.44,
#        "food_neg": 0.42,
#        "food_pos": 0.52,
#        "price_neg": 0.44,
#        "price_pos": 0.64,
#        "quality_neg": 0.54,
#        "quality_pos": 0.5,
#        "restaurant_neg": 0.25,
#        "restaurant_pos": 0.32,
#        "service_neg": 0.5,
#        "service_pos": 0.51
#    }

decision_dict = {
        "ambience_neg": 0.4,
        "ambience_pos": 0.4,
        "drinks_neg": 0.4,
        "drinks_pos": 0.4,
        "food_neg": 0.4,
        "food_pos": 0.4,
        "price_neg": 0.4,
        "price_pos": 0.4,
        "quality_neg": 0.4,
        "quality_pos": 0.4,
        "restaurant_neg": 0.4,
        "restaurant_pos": 0.4,
        "service_neg": 0.4,
        "service_pos": 0.4
    }


# for sensitive: reduce threshold by 20%
#for key in decision_dict.keys():
#    if "_neg" in key:
#        decision_dict[key] = decision_dict[key] * 0.6
#    else:
#        decision_dict[key] = decision_dict[key] * 0.8

if __name__ == "__main__":

    filename = "Full_Competitor_Reviews_translated.csv"
    data_columns = "text_german"

    with open(r"config_experimentation.json") as f:
        args = json.load(f)

    tokenizer_model = args["tokenizer_model"].split("-")
    if "%" in args["tokenizer_model"][0]:
        preperation_technique = args["tokenizer_model"][0].split("%")[0]
        preperation_ngram = args["tokenizer_model"][0].split("%")[1]
    else:
        preperation_technique = args["tokenizer_model"][0]
        preperation_ngram = 0

    if "%" in args["tokenizer_model"][1]:
        use_bin_dict = {"Binary": True, "Multi": False}
        model_technique = args["tokenizer_model"][1].split("%")[0]
        model_useBinary = use_bin_dict[args["tokenizer_model"][1].split("%")[0]]
    else:
        model_technique = args["tokenizer_model"][1]
        model_useBinary = False

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("prediction will be done on {}", device)

    ## do the data loading

    # do preprocessing
    ### train
    predict_df = pd_load_multiple_files(os.path.join(args["data_path"], "predict", filename), encoding="utf-8")
    predict_df = predict_df.dropna(subset= [data_columns], axis=0)

    # split into sentences
    print("Split sentences")
    ##currently not used
    #def spacy_sent_tokenize(text):
    #    doc = nlp(text)
    #    return [sent.string.strip() for sent in doc.sents]

    #time.sleep(0.5)
    predict_df[data_columns] = predict_df[data_columns].progress_apply(lambda x: sent_tokenize(x))
    predict_df = predict_df.explode(data_columns)
    predict_df = predict_df.reset_index(drop= True)
    predict_df = predict_df.reset_index(drop=False)

    ## do the preprocessing
    print("Preprocess")
    preprocessor = Preprocessor(doLower= args["doLower"], doLemmatization= args["doLemmatization"], removeStopWords= args["removeStopWords"], doSpellingCorrection= args["doSpellingCorrection"], removeNewLine= args["removeNewLine"], removePunctuation=args["removePunctuation"], removeHtmlTags= args["removeHtmlTags"], minTextLength = args["minTextLength"])
    predict_df["processed"] = preprocessor.fit_transform(predict_df["text_german"])
    predict_df = predict_df.dropna(subset=["processed"], axis=0)

    print("Tokenize")
    tokenizer = Tokenizer(tokenizeStr= preperation_technique, ngram= preperation_ngram, fasttextFile= args["fasttext_file"], doLower= args["doLower"])
    predict_df["processed"] = tokenizer.fit_transform(predict_df["processed"])

    ## for testing purposes
    #train_df = train_df.sample(100)
    #val_df = val_df.sample(20)
    #test_df = test_df.sample(20)

    ## apply the model
    labels = ["price_pos", "price_neg", "quality_pos", "quality_neg", "restaurant_pos", "restaurant_neg", "food_pos", "food_neg", "drinks_pos", "drinks_neg", "ambience_pos",
       "ambience_neg", "service_pos", "service_neg"]
    sentimentDict = {"pos": "positiv", "neu": "neutral", "neg": "negativ", "con": "uneinig"}
    cathegoryDict = {"price": "des Preises", "quality": "der Qualität", "restaurant": "des Restaurants", "food": "des Essens", "drinks": "der Getränke", "ambience": "des Ambientes", "service": "des Service"}
    texts = ["Die Bewertung {} ist {}.".format(cathegoryDict[x.split("_")[0]], sentimentDict[x.split("_")[1]]) for x in labels]
    labelSentencesDict = dict(zip(labels, texts))
    max_label_len = max([len(word_tokenize(x)) for x in labelSentencesDict.values()])

    print("Make Predictions")
    model = Model(binaryClassification= model_useBinary, model_str= model_technique, doLower= args["doLower"], train_batchSize= args["train_batchSize"], testval_batchSize= args["testval_batchSize"], learningRate= args["learningRate"], doLearningRateScheduler= args["doLearningRateScheduler"], labelSentences= labelSentencesDict, smartBatching=args["smartBatching"], max_label_len= max_label_len, device= device)

    model.load(os.path.join(args["model_path"], "apple-flambee-545.pt"))

    pred = model.predict(data=predict_df["processed"], device= device)

    pd.concat((predict_df.reset_index(drop= True), pred), axis= 1).to_csv(os.path.join(args["data_path"], "predict", filename[:-4] + "_predictions_raw.csv"))

    def logits_to_pred(column):
        thrshld = decision_dict[column.name]
        return (column > thrshld).astype(int)

    pred = pd.concat((pred, pred.apply(logits_to_pred, axis= 0).add_suffix("_pred")), axis=1)

    for category in ["ambience", "drinks", "food", "price", "quality", "restaurant", "service"]:
        pred[category] = np.nan
        pred.loc[(pred["{}_pos_pred".format(category)] != pred["{}_neg_pred".format(category)]) & (pred["{}_pos_pred".format(category)] == 1), category] = "positive"
        pred.loc[(pred["{}_pos_pred".format(category)] != pred["{}_neg_pred".format(category)]) & (pred["{}_pos_pred".format(category)] == 0), category] = "negative"
        pred.loc[(pred["{}_pos_pred".format(category)] == pred["{}_neg_pred".format(category)]) & (pred["{}_pos_pred".format(category)] == 0), category] = "neutral"
        pred.loc[(pred["{}_pos_pred".format(category)] == pred["{}_neg_pred".format(category)]) & (pred["{}_pos_pred".format(category)] == 1), category] = "conflict"

    export_df = pd.concat((predict_df.reset_index(drop= True), pred), axis=1)
    # drop unnecessary Unnamed columns
    export_df = export_df.rename({"Unnamed: 0": "review_id"}, axis= 1)
    export_df = export_df.drop([x for x in export_df.columns if "Unnamed" in x], axis= 1)

    export_df.to_csv(os.path.join(args["data_path"], "predict", filename[:-4] + "_predictions.csv"))

    ind_vars = ["name", "review_id", "reviewRating", "Review Date", "reviewUrl", "text_german", "lang", "address"]
    pred_vars = ['ambience', 'drinks', 'food', 'price', 'quality', 'restaurant', 'service']
    export_df = export_df.drop(list(set(export_df.columns) - set(ind_vars + pred_vars)), axis= 1)
    export_df = export_df.melt(id_vars= ind_vars, var_name="type", value_name="value")
    export_df.to_csv(os.path.join(args["data_path"], "predict", filename[:-4] + "_predictions_melted.csv"))