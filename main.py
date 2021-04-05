import json
import wandb
import logging
import os
import sys
import random

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch import optim

from preprocessing import Preprocessor
from tokenization import Tokenizer

from modeling import Model

from nltk import word_tokenize
import nltk
nltk.download('punkt')


def pd_load_multiple_files(path):
    if path.split(".")[-1] == "csv":
        return(pd.read_csv(path))
    elif (path.split(".")[-1] == "xlsx") or (path.split()[-1] == "xls"):
        return(pd.read_excel(path))
    elif path.split(".")[-1] == "pkl":
        return(pd.read_pickle(path))
    else:
        logging.error("{} datatype not supported.".format(path))
        sys.exit("{} datatype not supported.".format(path))


if __name__ == "__main__":
    with open(r"config.json") as f:
        args = json.load(f)

    wandb.init(project= "NewsClassification", entity='lexitech', config=args)
    args = wandb.config
    wandb.log({'finished': False})

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_model = json.loads(args["tokenizer_model"].replace("'", '"'))
    for dict_elements in tokenizer_model.keys():
        if tokenizer_model[dict_elements] == "True":
            tokenizer_model[dict_elements] = True
        elif tokenizer_model[dict_elements] == "False":
            tokenizer_model[dict_elements] = False
        else:
            pass

    logging.info("training will be done on {}", device)

    ## do the data loading
    ### check for already preprocessed files
    train_pre_path = os.path.join(args["data_path"], "temp", "{}_prep_train_{}-{}-{}-{}-{}_{}.pkl".format(args["train_data_file"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))
    val_pre_path = os.path.join(args["data_path"], "temp", "{}_prep_val_{}-{}-{}-{}-{}_{}.pkl".format(args["train_data_file"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))
    if args["test_data_file"]:
        test_pre_path = os.path.join(args["data_path"], "temp", "{}_prep_test_{}-{}-{}-{}-{}_{}.pkl".format(args["test_data_file"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))
    else:
        test_pre_path = os.path.join(args["data_path"], "temp", "{}_prep_test_{}-{}-{}-{}-{}_{}.pkl".format(args["train_data_file"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))
    ### check for already tokenized files
    if "ngram" in tokenizer_model.keys():
        train_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_train_{}-{}-{}-{}-{}-{}-{}_{}.pkl".format(args["train_data_file"], tokenizer_model["tokenizer"], tokenizer_model["ngram"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))
        val_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_val_{}-{}-{}-{}-{}-{}_{}-{}.pkl".format(args["train_data_file"], tokenizer_model["tokenizer"], tokenizer_model["ngram"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))
        if args["test_data_file"]:
            test_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_test_{}-{}-{}-{}-{}-{}-{}_{}.pkl".format(args["test_data_file"], tokenizer_model["tokenizer"], tokenizer_model["ngram"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))
        else:
            test_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_test_{}-{}-{}-{}-{}-{}-{}_{}.pkl".format(args["train_data_file"], tokenizer_model["tokenizer"], tokenizer_model["ngram"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))
    else:
        train_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_train_{}-{}-{}-{}-{}-{}_{}.pkl".format(args["train_data_file"], tokenizer_model["tokenizer"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))
        val_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_val_{}-{}-{}-{}-{}-{}_{}.pkl".format(args["train_data_file"], tokenizer_model["tokenizer"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))
        if args["test_data_file"]:
            test_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_test_{}-{}-{}-{}-{}-{}_{}.pkl".format(args["test_data_file"], tokenizer_model["tokenizer"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))
        else:
            test_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_test_{}-{}-{}-{}-{}-{}_{}.pkl".format(args["train_data_file"], tokenizer_model["tokenizer"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]))

    # TODO: Implement Whoosh Index for file storage store metadata with idx
    # TODO: Implement paraallel processing with pandarallel or dask

    ## reload data if existent
    run_preprocessing = False
    run_tokenization = False

    if os.path.exists(train_tok_path) and os.path.exists(val_tok_path) and os.path.exists(test_tok_path):
        df_train = pd.read_pickle(train_tok_path)
        df_val = pd.read_pickle(val_tok_path)
        df_test = pd.read_pickle(test_tok_path)
    elif os.path.exists(train_pre_path) and os.path.exists(val_pre_path) and os.path.exists(test_pre_path):
        df_train = pd.read_pickle(train_pre_path)
        df_val = pd.read_pickle(val_pre_path)
        df_test = pd.read_pickle(test_pre_path)
        run_tokenization = True
    else:
        run_preprocessing = True
        run_tokenization = True

    if run_preprocessing:
        ### train
        train_df = pd_load_multiple_files(os.path.join(args["data_path"], args["train_data_file"]))
        train_df = train_df.drop(args["train_data_drop"], axis= 1)
        train_df = train_df.sample(int(train_df.shape[0] * args["data_used"]))
        if args["train_target_file"] and args["train_merge_on"]:
            train_target = pd_load_multiple_files(os.path.join(args["data_path"], args["train_target_file"]))
            train_target = train_target.loc[:, args["targets"] + [args["train_merge_on"][1]]]
            train_df = pd.merge(train_df, train_target, how="left", left_on=args["train_merge_on"][0], right_on=args["train_merge_on"][1])
            train_df = train_df.drop([args["train_merge_on"][0]], axis= 1)
        if not train_df.shape[1] == len(args["targets"]) +1:
            logging.error("train_df has too many columns, check your files.")
            sys.exit("train_df has too many columns, check your files.")

        ### test
        if args["test_data_file"]:
            test_df = pd_load_multiple_files(os.path.join(args["data_path"], args["test_data_file"]))
            test_df = test_df.drop(args["test_data_drop"], axis=1)
            if args["test_target_file"] and args["test_merge_on"]:
                test_target = pd_load_multiple_files(os.path.join(args["data_path"], args["test_target_file"]))
                test_target = test_target.loc[:, args["targets"] + [args["test_merge_on"][1]]]
                test_df = pd.merge(test_df,  test_target, how="left", left_on=args["test_merge_on"][0], right_on=args["test_merge_on"][1])
                test_df =  test_df.drop([args["test_merge_on"][0]], axis=1)
            if not train_df.shape[1] == len(args["targets"]) + 1:
                logging.error("test_df has too many columns, check your files.")
                sys.exit("test_df has too many columns, check your files.")
        else:
            train_df, test_df = train_test_split(train_df, test_size= args["test_split"], random_state= 42)

        ### validation
        if args["validation_split"]:
            train_df, val_df = train_test_split(train_df, test_size=args["validation_split"], random_state=42)
        else:
            logging.error("vaidation_split needs to be given.")
            sys.exit("vaidation_split needs to be given.")

        ## get data and train columns
        data_column = list(set(train_df.columns) - set(args["targets"]))[0]

        ## do the preprocessing
        print("Preprocess")
        preprocessor = Preprocessor(doLower= args["doLower"], doLemmatization= args["doLemmatization"], removeStopWords= args["removeStopWords"], doSpellingCorrection= False, removeNewLine= args["removeNewLine"], removePunctuation=args["removePunctuation"], removeHtmlTags= False, minTextLength = args["minTextLength"])
        train_df[data_column] = preprocessor.fit_transform(train_df[data_column])
        train_df = train_df.dropna()
        val_df[data_column] = preprocessor.transform(val_df[data_column])
        val_df = val_df.dropna()
        test_df[data_column] = preprocessor.transform(test_df[data_column])
        test_df = test_df.dropna()

        ## save the preprocessed data
        if not os.path.exists(os.path.join(args["data_path"], "temp")):
            os.makedirs(os.path.join(args["data_path"], "temp"))
        train_df.to_pickle(train_pre_path)
        val_df.to_pickle(val_pre_path)
        test_df.to_pickle(test_pre_path)
    else:
        train_df = pd.read_pickle(train_pre_path)
        val_df = pd.read_pickle(val_pre_path)
        test_df = pd.read_pickle(test_pre_path)
        ## get data and train columns
        data_column = list(set(train_df.columns) - set(args["targets"]))[0]


    if run_tokenization:
        ## do tokenization
        print("Tokenize")
        tokenizer = Tokenizer(args= tokenizer_model, fasttextFile= args["fasttext_file"], doLower= args["doLower"])
        train_df[data_column] = tokenizer.fit_transform(train_df[data_column])
        val_df[data_column] = tokenizer.transform(val_df[data_column])
        test_df[data_column] = tokenizer.transform(test_df[data_column])

        ## save the preprocessed data
        if not os.path.exists(os.path.join(args["data_path"], "temp")):
            os.makedirs(os.path.join(args["data_path"], "temp"))
        train_df.to_pickle(train_tok_path)
        val_df.to_pickle(val_tok_path)
        test_df.to_pickle(test_tok_path)

    else:
        train_df = pd.read_pickle(train_tok_path)
        val_df = pd.read_pickle(val_tok_path)
        test_df = pd.read_pickle(test_tok_path)


    ## for testing purposes
    #train_df = train_df.sample(100)
    #val_df = val_df.sample(20)
    #test_df = test_df.sample(20)

    ## apply the model
    cathegoryDict = {"science_int": "science", "sports_int": "sports", "world_int": "the world", "business_int": "business"}
    texts = ["The article is about {}.".format(cathegoryDict[x]) for x in args["targets"]]
    labelSentencesDict = dict(zip(args["targets"], texts))
    max_label_len = max([len(word_tokenize(x)) for x in labelSentencesDict.values()])

    print("Train Model")
    model = Model(args= tokenizer_model, doLower= args["doLower"], train_batchSize= args["train_batchSize"], testval_batchSize= args["testval_batchSize"], learningRate= args["learningRate"], doLearningRateScheduler= args["doLearningRateScheduler"], labelSentences= labelSentencesDict, smartBatching=args["smartBatching"], max_label_len= max_label_len, device= device)

    model.run(train_data= train_df[data_column], train_target= train_df[args["targets"]], val_data= val_df[data_column], val_target= val_df[args["targets"]], test_data= test_df[data_column], test_target= test_df[args["targets"]], epochs= args["numEpochs"])

    wandb.log({'finished': True})

    #save the model
    #model.save(os.path.join(args["model_path"], "{}".format(wandb.run.name)))
