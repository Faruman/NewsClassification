import wandb
import math
import random
import time
import csv

import logging
import sys
import statistics
from tqdm import tqdm
from tqdm.contrib import tzip

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCELoss, CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch import optim

from transformers import DistilBertForSequenceClassification, BertForSequenceClassification, XLNetForSequenceClassification, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer
from transformers import get_cosine_schedule_with_warmup

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

import pickle

# function to plot precision/recall to Threshold graph
def plot_auc(label, score, title):
    precision, recall, thresholds = precision_recall_curve(label, score)
    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.plot(thresholds, precision[1:], color='r', label='Precision')
    plt.plot(thresholds, recall[1:], color='b', label='Recall')
    plt.gca().invert_xaxis()
    plt.legend(loc='lower right')

    plt.xlabel('Threshold (0.00 - 1.00)')
    plt.ylabel('Precision / Recall')
    _ = plt.title(title)
    return plt

#TODO: implement new step variabels

class Model():
    def __init__(self, args: dict, doLower: bool, train_batchSize: int, testval_batchSize:int, learningRate: float, doLearningRateScheduler: bool, target_columns: list, smartBatching: bool = True, mixedPrecision: bool = True, labelSentences: dict = None, max_label_len= None, model= None, optimizer= None, loss_fct= None, device= "cpu"):
        self.args = args
        self.labelSentences = labelSentences
        self.tokenizer = None
        self.device = device
        self.train_batchSize = train_batchSize
        self.testval_batchSize = testval_batchSize
        self.learningRate = learningRate
        self.optimizer = optimizer
        self.doLearningRateScheduler = doLearningRateScheduler
        self.learningRateScheduler = None
        self.smartBatching = smartBatching
        self.mixedPrecision = mixedPrecision
        self.max_label_len = max_label_len
        self.target_columns = target_columns
        self.input_multiclass_as_one = False

        if self.args["model"] in ["distilbert", "bert", "xlnet", "lstm", "roberta", "distilroberta"]:
            if loss_fct:
                self.loss_fct = loss_fct
            else:
                self.loss_fct = BCEWithLogitsLoss()

            if self.args["binaryClassification"]:
                self.num_labels = 1
            else:
                self.num_labels = len(self.labelSentences.keys())

        if self.args["model"] == "distilbert":
            if doLower:
                # distilbert german uncased should be used, however a pretrained model does not exist
                self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False, torchscript=True)
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            else:
                self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False, torchscript=True)
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

        elif self.args["model"] == "bert":
            if doLower:
                self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False, torchscript=True)
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            else:
                self.model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False, torchscript=True)
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        elif self.args["model"] == "xlnet":
            if doLower:
                self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False, torchscript=True)
                self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            else:
                self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False, torchscript=True)
                self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

        elif self.args["model"] == "roberta":
            if doLower:
                self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False, torchscript=True)
                self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            else:
                self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False, torchscript=True)
                self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        elif self.args["model"] == "distilroberta":
            if doLower:
                self.model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False, torchscript=True)
                self.tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
            else:
                self.model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=self.num_labels, output_attentions=False, output_hidden_states=False, torchscript=True)
                self.tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

        #elif self.args["model"] == "CNN":
        #    self.model = MyLSTM(num_labels=self.num_labels)

        elif self.args["model"] == "gradientboost":
            self.model = GradientBoostingClassifier(learning_rate= self.learningRate, n_estimators= self.args["n_estimators"], max_depth= self.args["max_depth"], verbose=1)
            self.input_multiclass_as_one = True

        elif self.args["model"] == "randomforest":
            self.model = RandomForestClassifier(learning_rate= self.learningRate, n_estimators= self.args["n_estimators"], max_depth= self.args["max_depth"], verbose=1, n_jobs= -1)
            self.input_multiclass_as_one = True

        elif self.args["model"] == "naivebayes":
            self.model = OneVsRestClassifier(MultinomialNB(alpha= self.learningRate))

        elif self.args["model"] == "naivebayes_norm":
            self.model = Pipeline([
                ("nb_norm", MinMaxScaler()),
                ("nb_clf", OneVsRestClassifier(MultinomialNB(alpha= self.learningRate)))
                ])

        elif self.args["model"] == "sgd":
            self.model = OneVsRestClassifier(SGDClassifier(alpha= self.learningRate, loss='hinge', penalty='l2'))

        else:
            logging.error("Define a model in the args dict.")
            sys.exit("Define a model in the args dict.")

    def preprocess(self, data, target, max_label_len, target_columns):
        if self.args["model"] in ["distilbert", "bert", "xlnet", "roberta", "distilroberta"]:
            df = pd.DataFrame([[a, b] for a, b in data], columns=["data", "mask"])
            df = pd.concat([df, pd.DataFrame(target, columns=target_columns)], axis=1)
            if self.args["binaryClassification"]:
                max_label_len += 2
                if type(self.labelSentences[list(self.labelSentences.keys())[0]]) == str:
                    for key in self.labelSentences.keys():
                        text = self.labelSentences[key]
                        temp = self.tokenizer(text, return_attention_mask=True, padding="max_length", truncation=True, max_length= max_label_len)
                        encoded_text = temp["input_ids"][1:]
                        mask = temp["attention_mask"][1:]
                        self.labelSentences[key] = (encoded_text, mask)
                max_label_len -= 1
                if set(target_columns).issubset(set(self.labelSentences.keys())):
                    def create_samples(df_row, target_columns):
                        output_base = list()
                        output_mask = list()
                        for i, key in enumerate(target_columns):
                            input_base = df_row["data"].copy()
                            input_mask = df_row["mask"].copy()
                            extend_text, extend_mask = self.labelSentences[key]
                            last_data = np.max(np.nonzero(input_mask)) +1
                            if last_data < (len(input_mask)- len(extend_mask)):
                                input_base[last_data: (last_data+ len(extend_mask))] = extend_text
                                input_mask[last_data: (last_data+ len(extend_mask))] = extend_mask
                                output_base.append(input_base)
                                output_mask.append(input_mask)
                            else:
                                input_base[-(len(extend_text)+1):] = [input_base[last_data -1]] + extend_text
                                input_mask[-(len(extend_mask)+1):] = [input_base[last_data -1]] + extend_mask
                                output_base.append(input_base)
                                output_mask.append(input_mask)
                        df_row["data"] = np.array(output_base)
                        df_row["mask"] = np.array(output_mask)
                        return df_row
                    df = df.apply(create_samples, args= (target_columns,), axis=1)
                else:
                    logging.error("Target columns need to be subset of labelSentences.keys.")
                    sys.exit("Target columns need to be subset of labelSentences.keys.")
                return df["data"].values, df["mask"].values, target
            else:
                return df["data"].values, df["mask"].values, target
        else:
            mask = np.full(data.shape, 1)
            return data, mask, target

    def applySmartBatching(self, data, mask, target= None, index= None, text= "Iteration:"):
        data = np.stack(data)
        mask = np.stack(mask)
        if target is not None and index is None:
            target = target
        elif index is not None and target is None:
            index = index
        else:
            logging.warning("Provide exactly one of target or index.")

        def getArrayLength(x):
            return sum(x != 0)

        length_array = np.apply_along_axis(getArrayLength, np.stack(data).ndim - 1, np.stack(data))
        while length_array.ndim > 1:
            length_array = np.max(length_array, axis=1)
        sort_idx = length_array.argsort()
        length_array = length_array[sort_idx]
        data = data[sort_idx]
        mask = mask[sort_idx]
        if target is not None and index is None:
            target = target[sort_idx]
        elif index is not None and target is None:
            index = index[sort_idx]
        else:
            logging.warning("Provide exactly one of target or index.")

        data_batch = list()
        mask_batch = list()
        if target is not None and index is None:
            target_batch = list()
        elif index is not None and target is None:
            index_batch = list()
        else:
            logging.warning("Provide exactly one of target or index.")

        pbar = tqdm(total=len(data), desc="Apply dynamic batching")
        while len(data) > 0:
            to_take = min(self.train_batchSize, len(data))
            select = random.randint(0, len(data) - to_take)
            max_batch_len = max(length_array[select:select + to_take])
            data_batch += [torch.tensor(data[select:select + to_take][..., :max_batch_len], dtype=torch.long)]
            mask_batch += [torch.tensor(mask[select:select + to_take][..., :max_batch_len], dtype=torch.long)]
            if target is not None and index is None:
                target_batch += [torch.tensor(target[select:select + to_take], dtype=torch.long)]
            elif index is not None and target is None:
                index_batch += [torch.tensor(index[select:select + to_take], dtype=torch.long)]
            else:
                logging.error("Provide exactly one of target or index.")
            length_array = np.delete(length_array, np.s_[select:select + to_take], 0)
            data = np.delete(data, np.s_[select:select + to_take], 0)
            mask = np.delete(mask, np.s_[select:select + to_take], 0)
            if target is not None and index is None:
                target = np.delete(target, np.s_[select:select + to_take], 0)
            elif index is not None and target is None:
                index = np.delete(index, np.s_[select:select + to_take], 0)
            else:
                logging.warning("Provide exactly one of target or index.")
            pbar.update(to_take)
        pbar.close()
        if target is not None and index is None:
            return tzip(data_batch, mask_batch, target_batch, desc=text)
        elif index is not None and target is None:
            return tzip(data_batch, mask_batch, index_batch, desc=text)
        else:
            return tzip(data_batch, mask_batch, desc=text)

    def applyNormalBatching(self, data, mask, target = None, text= "Iteration:"):
        data = torch.tensor(np.stack(data), dtype=torch.long)
        mask = torch.tensor(np.stack(mask), dtype=torch.long)
        if target is not None:
            target = torch.tensor(target, dtype=torch.int32)
            data = TensorDataset(data, mask, target)
        else:
            data = TensorDataset(data, mask)
        return tqdm(DataLoader(data, batch_size=self.train_batchSize), text)

    def train(self, data, mask, target, device= "cpu"):
        # TODO: recreate batches each epoch? => no, create extra argument
        # TODO: Create own training routine for simple models
        if self.args["model"] in ["distilbert", "bert", "xlnet", "lstm", "roberta", "distilroberta"]:
            if self.smartBatching:
                dataloader = self.applySmartBatching(data, mask, target, text= "Do training:")
            else:
                dataloader = self.applyNormalBatching(data, mask, target, text= "Do Training:")

            self.model.train()

            for step, batch in enumerate(dataloader):
                # TODO: Make loss function variable
                batch = tuple(t.to(device) for t in batch)
                data, mask, target = batch

                self.optimizer.zero_grad()

                if self.args["binaryClassification"]:
                    data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
                    mask = mask.reshape(mask.shape[0]*mask.shape[1], mask.shape[2])
                    target = target.reshape(target.shape[0]*target.shape[1])

                    if self.args["model"] in ["distilbert", "bert", "xlnet", "roberta", "distilroberta"]:
                        data = torch.split(data, int(data.shape[0] / len(self.labelSentences.keys())))
                        mask = torch.split(mask, int(mask.shape[0] / len(self.labelSentences.keys())))
                        target = torch.split(target, int(target.shape[0] / len(self.labelSentences.keys())))

                        sum_loss = 0
                        for data_batch, mask_batch, target_batch in zip(data, mask, target):
                            logits = self.model(input_ids= data_batch, attention_mask= mask_batch)[0]
                            loss = self.loss_fct(logits.flatten(), target_batch.type_as(logits))
                            sum_loss += loss.item()
                            loss.backward()

                        wandb.log({'train_batch_loss': sum_loss})

                    else:
                        sum_loss = 0
                        for i, label in enumerate(self.target_columns):
                            model_output = self.model[label](input_ids=data, attention_mask=mask)
                            subtarget = target[:, i]
                            loss = self.loss_fct(model_output, subtarget)
                            sum_loss += loss.item()
                            loss.backward()

                        wandb.log({'train_batch_loss': sum_loss})

                else:
                    if self.args["model"] in ["distilbert", "bert", "xlnet", "roberta", "distilroberta"]:
                        logits = self.model(input_ids= data, attention_mask= mask)[0]

                        loss = self.loss_fct(logits, target.type_as(logits))

                        loss.backward()
                        wandb.log({'train_batch_loss': loss.item()})

                    else:
                        model_output = self.model(input_ids=data, attention_mask=mask)
                        loss = self.loss_fct(model_output, target)

                        loss.backward()
                        wandb.log({'train_batch_loss': loss.item()})

                self.optimizer.step()

                if self.learningRateScheduler:
                    self.learningRateScheduler.step()
        else:
            if self.input_multiclass_as_one:
                self.model.fit(data, np.argmax(target, axis=1))
            else:
                self.model.fit(data, target)


    def test_validate(self, data, mask, target, type: str, device= "cpu", use_wandb= True, decision_dict= None):
        if not decision_dict:
            decision_dict = dict(zip(self.target_columns, [0.5]*len(self.target_columns)))

        if self.args["model"] in ["distilbert", "bert", "xlnet", "lstm", "roberta", "distilroberta"]:
            if self.smartBatching:
                dataloader = self.applySmartBatching(data, mask, target, text= "Do {}:".format(type))
            else:
                dataloader = self.applyNormalBatching(data, mask, target, text= "Do {}:".format(type))

            if self.args["model"] in ["distilbert", "bert", "xlnet", "roberta", "distilroberta"]:
                self.model.eval()
            else:
                if self.args["binaryClassification"]:
                    for key in self.model:
                        self.model[key] = self.model[key].eval()
                    else:
                        self.model = self.model.eval()

            all_model_outputs= []
            all_targets = []

            with torch.no_grad():
                for step, batch in enumerate(dataloader):
                    data, mask, target = batch
                    data = data.to(device)
                    mask = mask.to(device)

                    if self.args["binaryClassification"]:
                        if self.args["model"] in ["distilbert", "bert", "xlnet", "roberta", "distilroberta"]:
                            model_output = []
                            for i, label in enumerate(self.target_columns):
                                ind_model_output = self.model(data[:, i, :], mask[:, i, :])[0]
                                model_output.append(ind_model_output)
                            model_output = torch.sigmoid(torch.cat(model_output, 1))

                        else:
                            model_output = []
                            for i, label in enumerate(self.target_columns):
                                ind_model_output = self.model[label](data, mask)
                                model_output.append(ind_model_output)
                            model_output = torch.sigmoid(torch.cat(model_output, 0))
                    else:
                        if self.args["model"] in ["distilbert", "bert", "xlnet", "roberta", "distilroberta"]:
                            model_output = self.model(data, mask)[0]
                            model_output = torch.sigmoid(model_output)

                        else:
                            model_output = self.model(data, mask)
                            model_output = torch.sigmoid(model_output)

                    all_model_outputs.append(model_output.detach().cpu().numpy())
                    all_targets.append(target)

            all_targets = np.concatenate(all_targets)
            all_model_outputs = np.concatenate(all_model_outputs)
            all_model_outputs.transpose()
        else:
            all_targets = target
            all_model_outputs = self.model.predict_proba(data)

        all_model_outputs = np.argmax(all_model_outputs, axis=1)
        all_targets = np.argmax(all_targets, axis=1)

        macroF1 = f1_score(all_targets, all_model_outputs, average= "macro")
        macroPrec = precision_score(all_targets, all_model_outputs, average= "macro")
        macroRec = recall_score(all_targets, all_model_outputs, average= "macro")
        Acc = accuracy_score(all_targets, all_model_outputs)

        if use_wandb:
            wandb.log({'{}_macroF1'.format(type): macroF1, '{}_macroPrec'.format(type): macroPrec, '{}_macroRec'.format(type): macroRec, '{}_Acc'.format(type): Acc})
        else:
            return {'{}_macroF1'.format(type): macroF1, '{}_macroPrec'.format(type): macroPrec, '{}_macroRec'.format(type): macroRec, '{}_Acc'.format(type): Acc}

    def run(self, train_data, train_target, val_data, val_target, test_data, test_target, epochs: int):
        train_data, train_mask, train_target = self.preprocess(train_data, train_target, self.max_label_len, self.target_columns)
        val_data, val_mask, val_target = self.preprocess(val_data, val_target, self.max_label_len, self.target_columns)
        test_data, test_mask, test_target = self.preprocess(test_data, test_target, self.max_label_len, self.target_columns)

        if self.args["model"] in ["distilbert", "bert", "xlnet", "lstm", "roberta", "distilroberta"]:
            if self.args["optimizer"] == "adam":
                self.optimizer = optim.Adam(self.model.parameters(), self.learningRate)
            elif self.args["optimizer"] == "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), self.learningRate)
            else:
                self.optimizer = None

            if not self.optimizer:
                self.optimizer = optim.Adam(self.model.parameters(), self.learningRate)
            if ~bool(self.learningRateScheduler) and self.doLearningRateScheduler:
                num_train_steps = epochs * math.ceil(train_data.shape[0] / self.train_batchSize)
                self.learningRateScheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=int(0.1*num_train_steps), num_training_steps=num_train_steps)

            self.model.to(self.device)

            for i in range(epochs):
                print("epoch {}".format(i))
                self.train(train_data, train_mask, train_target, device= self.device)
                self.test_validate(val_data, val_mask, val_target, type= "validate", device= self.device)
            self.test_validate(test_data, test_mask, test_target, type= "test", device= self.device)

        else:
            self.train(train_data, train_mask, train_target, device=self.device)
            self.test_validate(val_data, val_mask, val_target, type="validate", device=self.device)
            self.test_validate(test_data, test_mask, test_target, type="test", device=self.device)

    def save(self, file_path: str):
        if self.args["model"] in ["distilbert", "bert", "xlnet", "lstm", "roberta", "distilroberta"]:
            # save as torchscript
            tokens = self.tokenizer("All the appetizers and salads were fabulous, the steak was mouth watering and the pasta was delicious!!! [SEP] Die Bewertung des Preises ist positiv.", padding="max_length", max_length=512)
            mask = torch.tensor([tokens["attention_mask"]]).to("cuda")
            tokens = torch.tensor([tokens["input_ids"]]).to("cuda")
            self.model.eval()
            traced_model = torch.jit.trace(self.model, (tokens, mask))
            traced_model.save(file_path + ".pt")
        else:
            with open(file_path + ".pkl", 'wb') as file:
                pickle.dump(self.model, file)

        pd.DataFrame(data=self.target_columns, columns=["target"]).to_csv(file_path[:-3] + "_targetConfig.csv")


    def load(self, file_path):
        if self.args["model"] in ["distilbert", "bert", "xlnet", "lstm", "roberta", "distilroberta"]:
            self.model = torch.jit.load(file_path)
        else:
            with open(file_path, 'rb') as file:
                self.model = pickle.load(file)

        self.target_columns = list(pd.read_csv(file_path[:-3] + "_targetConfig.csv")["target"])

    def predict(self, data, device="cpu"):
        # Fake target system
        target = pd.DataFrame(data= np.zeros((data.shape[0], len(self.target_columns))), columns=self.target_columns)

        data, mask, target = self.preprocess(data, target, self.max_label_len)

        if self.args["model"] in ["distilbert", "bert", "xlnet", "lstm", "roberta", "distilroberta"]:
            start_index = pd.DataFrame(data= range(data.shape[0]), columns=["index"])
            if self.smartBatching:
                dataloader = self.applySmartBatching(data, mask, index= start_index, text="Do Inference")
            else:
                dataloader = self.applyNormalBatching(data, mask, text="Do Inference")

            if self.args["model"] in ["distilbert", "bert", "xlnet", "roberta", "distilroberta"]:
                self.model.eval()
            else:
                if self.args["binaryClassification"]:
                    for key in self.model:
                        self.model[key] = self.model[key].eval()
                    else:
                        self.model = self.model.eval()

            all_model_outputs = []
            all_index = []

            with torch.no_grad():
                for step, batch in enumerate(dataloader):
                    data, mask, index = batch
                    data = data.to(device)
                    mask = mask.to(device)

                    if self.args["binaryClassification"]:
                        if self.args["model"] in ["distilbert", "bert", "xlnet", "roberta", "distilroberta"]:
                            model_output = []
                            for i, label in enumerate(self.target_columns):
                                ind_model_output = self.model(data[:, i, :], mask[:, i, :])[0]
                                model_output.append(ind_model_output)
                            model_output = torch.sigmoid(torch.cat(model_output, 1))

                        else:
                            model_output = []
                            for i, label in enumerate(self.target_columns):
                                ind_model_output = self.model[label](input_ids=data, attention_mask=mask)
                                model_output.append(ind_model_output)
                            model_output = torch.sigmoid(torch.cat(model_output, 0))
                    else:
                        if self.args["model"] in ["distilbert", "bert", "xlnet", "roberta", "distilroberta"]:
                            model_output = self.model(input_ids=data, attention_mask=mask)[0]
                            model_output = torch.sigmoid(model_output)

                        else:
                            model_output = self.model(input_ids=data, attention_mask=mask)
                            model_output = torch.sigmoid(model_output)

                    all_model_outputs.append(model_output.detach().cpu().numpy())
                    all_index.append(index)

            all_index = np.concatenate(all_index)
            all_model_outputs = np.concatenate(all_model_outputs)

            output = pd.DataFrame(index= all_index.flatten(), data= all_model_outputs, columns= self.target_columns)
            output = output.reindex(start_index["index"].values)

        else:
            output = self.model.predict(data)

        return output


    # TODO: implement variable loss functions from torch-optimizer package