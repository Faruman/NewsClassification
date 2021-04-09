import json
import wandb
import logging
import os
import sys
import random

import pandas as pd
import numpy as np
from scipy import sparse

from sklearn.model_selection import train_test_split

import torch
from torch import optim

from preprocessing import Preprocessor
from tokenization import Tokenizer

from modeling import Model

from nltk import word_tokenize
import nltk
nltk.download('punkt')

import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "SentencePiece"])

# implementation of a dataloader which can handle multiple file types
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
    # get all the configuration details from the config.json file
    with open(r"config.json") as f:
        args = json.load(f)

    # initalize wandb to have online login and be able to run sweeps
    wandb.init(project= "NewsClassification", entity='faruman', config=args)
    args = wandb.config
    wandb.log({'finished': False})

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # define on which device to run
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

    # do the data loading
    ## check for already preprocessed files
    train_pre_path = os.path.join(args["data_path"], "temp", "{}_prep_train_{}-{}-{}-{}-{}-{}".format(args["train_data_file"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")
    val_pre_path = os.path.join(args["data_path"], "temp", "{}_prep_val_{}-{}-{}-{}-{}-{}".format(args["train_data_file"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")
    if args["test_data_file"]:
        test_pre_path = os.path.join(args["data_path"], "temp", "{}_prep_test_{}-{}-{}-{}-{}-{}".format(args["test_data_file"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")
    else:
        test_pre_path = os.path.join(args["data_path"], "temp", "{}_prep_test_{}-{}-{}-{}-{}-{}".format(args["train_data_file"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")
    ## check for already tokenized files
    if "ngram" in tokenizer_model.keys():
        train_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_train_{}-{}-{}-{}-{}-{}-{}-{}".format(args["train_data_file"], tokenizer_model["tokenizer"], tokenizer_model["ngram"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")
        val_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_val_{}-{}-{}-{}-{}-{}-{}_{}".format(args["train_data_file"], tokenizer_model["tokenizer"], tokenizer_model["ngram"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")
        if args["test_data_file"]:
            test_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_test_{}-{}-{}-{}-{}-{}-{}-{}".format(args["test_data_file"], tokenizer_model["tokenizer"], tokenizer_model["ngram"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")
        else:
            test_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_test_{}-{}-{}-{}-{}-{}-{}-{}".format(args["train_data_file"], tokenizer_model["tokenizer"], tokenizer_model["ngram"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")
    else:
        train_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_train_{}-{}-{}-{}-{}-{}-{}".format(args["train_data_file"], tokenizer_model["tokenizer"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")
        val_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_val_{}-{}-{}-{}-{}-{}-{}".format(args["train_data_file"], tokenizer_model["tokenizer"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")
        if args["test_data_file"]:
            test_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_test_{}-{}-{}-{}-{}-{}-{}".format(args["test_data_file"], tokenizer_model["tokenizer"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")
        else:
            test_tok_path = os.path.join(args["data_path"], "temp", "{}_tok_test_{}-{}-{}-{}-{}-{}-{}".format(args["train_data_file"], tokenizer_model["tokenizer"], args["doLower"], args["doLemmatization"], args["removeStopWords"], args["removeNewLine"], args["removePunctuation"], args["data_used"]) + "_{}")

    # TODO: Implement Whoosh Index for file storage store metadata with idx
    # TODO: Implement paraallel processing with pandarallel or dask

    ## reload data if existent
    run_preprocessing = False
    run_tokenization = False

    if (os.path.exists(train_tok_path.format("data") + ".npy") and os.path.exists(train_tok_path.format("target") + ".npy") and os.path.exists(val_tok_path.format("data") + ".npy") and os.path.exists(val_tok_path.format("target") + ".npy") and os.path.exists(test_tok_path.format("data") + ".npy") and os.path.exists(test_tok_path.format("target") + ".npy")) or (os.path.exists(train_tok_path.format("data") + ".npz") and os.path.exists(train_tok_path.format("target") + ".npy") and os.path.exists(val_tok_path.format("data") + ".npz") and os.path.exists(val_tok_path.format("target") + ".npy") and os.path.exists(test_tok_path.format("data") + ".npz") and os.path.exists(test_tok_path.format("target") + ".npy")):
        if "bow-" in train_tok_path or "tfidf-" in train_tok_path:
            train_data = sparse.load_npz(train_tok_path.format("data") + ".npz")
        else:
            train_data = np.load(train_tok_path.format("data") + ".npy", allow_pickle=True)
        train_target = np.load(train_tok_path.format("target") + ".npy", allow_pickle=True)
        if "bow-" in val_tok_path or "tfidf-" in val_tok_path:
            val_data = sparse.load_npz(val_tok_path.format("data") + ".npz")
        else:
            val_data = np.load(val_tok_path.format("data") + ".npy", allow_pickle=True)
        val_target = np.load(val_tok_path.format("target") + ".npy", allow_pickle=True)
        if "bow-" in test_tok_path or "tfidf-" in test_tok_path:
            test_data = sparse.load_npz(test_tok_path.format("data") + ".npz")
        else:
            test_data = np.load(test_tok_path.format("data") + ".npy", allow_pickle=True)
        test_target = np.load(test_tok_path.format("target") + ".npy", allow_pickle=True)
    elif os.path.exists(train_pre_path.format("data") + ".npy") and os.path.exists(train_pre_path.format("target") + ".npy") and os.path.exists(val_pre_path.format("data") + ".npy") and os.path.exists(val_pre_path.format("target") + ".npy") and os.path.exists(test_pre_path.format("data") + ".npy") and os.path.exists(test_pre_path.format("target") + ".npy"):
        train_data = np.load(train_pre_path.format("data") + ".npy", allow_pickle=True)
        val_data = np.load(val_pre_path.format("data") + ".npy", allow_pickle=True)
        test_data = np.load(test_pre_path.format("data") + ".npy", allow_pickle=True)
        train_target = np.load(train_pre_path.format("target") + ".npy", allow_pickle=True)
        val_target = np.load(val_pre_path.format("target") + ".npy", allow_pickle=True)
        test_target = np.load(test_pre_path.format("target") + ".npy", allow_pickle=True)
        run_tokenization = True
    else:
        run_preprocessing = True
        run_tokenization = True

    if run_preprocessing:
        ## create the train data
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

        ## create the test data
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

        ## create the validation data
        if args["validation_split"]:
            train_df, val_df = train_test_split(train_df, test_size=args["validation_split"], random_state=42)
        else:
            logging.error("vaidation_split needs to be given.")
            sys.exit("vaidation_split needs to be given.")

        ## get data columns
        data_column = list(set(train_df.columns) - set(args["targets"]))[0]
        train_target = train_df[args["targets"]].values
        val_target = val_df[args["targets"]].values
        test_target = test_df[args["targets"]].values

        ## do the preprocessing
        print("Preprocess")
        preprocessor = Preprocessor(doLower= args["doLower"], doLemmatization= args["doLemmatization"], removeStopWords= args["removeStopWords"], doSpellingCorrection= False, removeNewLine= args["removeNewLine"], removePunctuation=args["removePunctuation"], removeHtmlTags= False, minTextLength = args["minTextLength"])
        train_data = preprocessor.fit_transform(train_df[data_column])
        train_target = train_target[~pd.isnull(train_data)]
        train_data = train_data[~pd.isnull(train_data)]
        val_data = preprocessor.transform(val_df[data_column])
        val_target = val_target[~pd.isnull(val_data)]
        val_data = val_data[~pd.isnull(val_data)]
        test_data = preprocessor.transform(test_df[data_column])
        test_target = test_target[~pd.isnull(test_data)]
        test_data = test_data[~pd.isnull(test_data)]

        ## save the preprocessed data
        if not os.path.exists(os.path.join(args["data_path"], "temp")):
            os.makedirs(os.path.join(args["data_path"], "temp"))
        np.save(train_pre_path.format("data"), train_data, allow_pickle=True)
        np.save(val_pre_path.format("data"), val_data, allow_pickle=True)
        np.save(test_pre_path.format("data"), test_data, allow_pickle=True)
        np.save(train_pre_path.format("target"), train_target, allow_pickle=True)
        np.save(val_pre_path.format("target"), val_target, allow_pickle=True)
        np.save(test_pre_path.format("target"), test_target, allow_pickle=True)


    if run_tokenization:
        ## do tokenization
        print("Tokenize")
        tokenizer = Tokenizer(args= tokenizer_model, fasttextFile= args["fasttext_file"], doLower= args["doLower"])
        train_data = tokenizer.fit_transform(train_data)
        val_data = tokenizer.transform(val_data)
        test_data = tokenizer.transform(test_data)

        ## save the preprocessed data
        if not os.path.exists(os.path.join(args["data_path"], "temp")):
            os.makedirs(os.path.join(args["data_path"], "temp"))
        if sparse.issparse(train_data):
            sparse.save_npz(train_tok_path.format("data"), train_data)
        else:
            np.save(train_tok_path.format("data"), train_data)
        np.save(train_tok_path.format("target"), train_target)
        if sparse.issparse(val_data):
            sparse.save_npz(val_tok_path.format("data"), val_data)
        else:
            np.save(val_tok_path.format("data"), val_data)
        np.save(val_tok_path.format("target"), val_target)
        if sparse.issparse(val_data):
            sparse.save_npz(test_tok_path.format("data"), test_data)
        else:
            np.save(test_tok_path.format("data"), test_data)
        np.save(test_tok_path.format("target"), test_target)

    ## for testing purposes
    #train_df = train_df.sample(100)
    #val_df = val_df.sample(20)
    #test_df = test_df.sample(20)

    # train the model
    cathegoryDict = {"science_int": "science", "sports_int": "sports", "world_int": "the world", "business_int": "business"}
    ## create the auxiliary sentences which are needed if binary classification is done.
    texts = ["The article is about {}.".format(cathegoryDict[x]) for x in args["targets"]]
    labelSentencesDict = dict(zip(args["targets"], texts))
    max_label_len = max([len(word_tokenize(x)) for x in labelSentencesDict.values()])

    print("Train Model")
    model = Model(args= tokenizer_model, doLower= args["doLower"], train_batchSize= args["train_batchSize"], testval_batchSize= args["testval_batchSize"], learningRate= args["learningRate"], doLearningRateScheduler= args["doLearningRateScheduler"], labelSentences= labelSentencesDict, smartBatching=args["smartBatching"], max_label_len= max_label_len, device= device, target_columns= args["targets"])

    # train and test the model
    model.run(train_data= train_data, train_target= train_target, val_data= val_data, val_target= val_target, test_data= test_data, test_target= test_target, epochs= args["numEpochs"])

    # close the logging
    wandb.log({'finished': True})

    # save the model
    #model.save(os.path.join(args["model_path"], "{}".format(wandb.run.name)))
