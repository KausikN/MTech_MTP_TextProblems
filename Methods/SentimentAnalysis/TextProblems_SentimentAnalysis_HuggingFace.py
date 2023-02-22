"""
Text Problems - Sentiment Analysis - HuggingFace

References:

"""

# Imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .Utils import *

# Main Classes
# HuggingFace - Dataset Loader
class DatasetLoader_SentimentAnalysis_HuggingFace(DatasetLoader_SentimentAnalysis_Base):
    def __init__(self, inputs, targets, tokenizer, max_len, **params):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.__dict__.update(params)
    
    def __getitem__(self, item):
        ## Init
        text = str(self.inputs[item])
        target = self.targets[item]
        ## Encode
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ## Input IDs
        input_ids = pad_sequences(
            encoding["input_ids"], 
            maxlen=self.max_len, dtype=torch.Tensor, truncating="post", padding="post"
        ).astype(dtype="int64")
        input_ids = torch.tensor(input_ids) 
        ## Attention Mask
        attention_mask = pad_sequences(
            encoding["attention_mask"], 
            maxlen=self.max_len, dtype=torch.Tensor, truncating="post", padding="post"
        ).astype(dtype="int64")
        attention_mask = torch.tensor(attention_mask)       

        return {
            "input": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask.flatten(),
            "targets": torch.tensor(target, dtype=torch.long)
        }
# HuggingFace - Model
class TextProblems_SentimentAnalysis_HuggingFace(TextProblems_SentimentAnalysis_Base):
    def __init__(self,
    n_classes=2,
    dataset_params={
        "test_size": 0.2,
        "val_size": 0.05,
        "random_state": 0,
        "max_len": 16,
        "batch_size": 4
    },
    train_params={
        "batch_size": 4,
        "epochs": 3,
        "learning_rate": 3e-5,
        "save_dir": "_models/temp/"
    },
    model_params={
        "model_id": "xlnet-base-cased",
        "load_pretrained": True
    },
    random_state=0,
    **params
    ):
        '''
        Text Problems - Sentiment Analysis - Hugging Face

        Params:
         - n_classes : Number of classes
         - dataset_params : Dataset Parameters
         - train_params : Training Parameters
         - model_params : Model Parameters
         - random_state : Random State

        '''
        self.n_classes = n_classes
        self.dataset_params = dataset_params
        self.train_params = train_params
        self.model_params = model_params
        self.random_state = random_state
        self.base_params = {
            "dataset_loader": DatasetLoader_SentimentAnalysis_HuggingFace
        }
        self.__dict__.update(params)
        # Time Params
        self.time_data = {
            "train": {},
            "predict": {}
        }
        # History
        self.history = {
            "history": [],
            "best_model_info": None
        }
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Features Info
        self.features_info = {
            "input": [],
            "target": {
                "name": "Sentiment",
                "type": {
                    "type": "category",
                    "categories": ["negative", "positive"]
                }
            }
        }
        # Model and Tokenizer
        self.tokenizer = None
        self.model = {"model": None}
        if self.model_params["load_pretrained"]: self.load_pretrained(self.model_params["model_id"])
        # Save/Load Params
        self.save_paths = {
            "class_obj": "class_obj.p",
            "model": "model.pt",
            "tokenizer": "tokenizer/"
        }

    def train(self,
        Fs, Ls, 
        features_info={"input": [], "target": {}},
        **params
        ):
        '''
        Train

        Train Model on given features and labels.

        Inputs:
            - Fs : Text Input (N_Samples, 1)
            - Ls : Label Distribution (N_Samples, Label_Dim)
            
        Outputs:
            - model : Model that can be used to predict labels from features
        '''
        # Init
        self.time_data["train"] = Time_Record("Model - Train")
        ## Features Info Init
        self.features_info = features_info
        ## Data Init
        Fs = np.array(Fs["input"])
        Ls = np.array(Ls)
        N_CLASSES = self.n_classes
        if Ls.shape[1] != N_CLASSES: raise Exception("Ls shape does not match n_classes")
        # Preprocess
        DATASETS = {"Fs": {}, "Ls": {}}
        DATASETS["Fs"]["train"], DATASETS["Fs"]["test"], DATASETS["Ls"]["train"], DATASETS["Ls"]["test"] = train_test_split(
            Fs, Ls, 
            test_size=self.dataset_params["test_size"], random_state=self.dataset_params["random_state"]
        )
        DATASETS["Fs"]["train"], DATASETS["Fs"]["val"], DATASETS["Ls"]["train"], DATASETS["Ls"]["val"] = train_test_split(
            DATASETS["Fs"]["train"], DATASETS["Ls"]["train"], 
            test_size=self.dataset_params["val_size"], random_state=self.dataset_params["random_state"]
        )
        DATASET_LOADERS = {}
        for dk in DATASETS["Fs"].keys():
            D = self.base_params["dataset_loader"](
                DATASETS["Fs"][dk], DATASETS["Ls"][dk], 
                self.tokenizer, self.dataset_params["max_len"]
            )
            DL = DataLoader(
                D, 
                batch_size=self.dataset_params["batch_size"],
                num_workers=4
            )
            DATASET_LOADERS[dk] = DL
        self.time_data["train"] = Time_Record("Data Preprocess", self.time_data["train"])
        # Train
        ## Init Model
        MODEL = self.model["model"]
        if self.model["model"] is None:
            self.load_pretrained(self.model_params["model_id"])
            MODEL = self.model["model"]
        MODEL = MODEL.to(self.device)
        ## Init Optimizer
        PARAM_OPTIM = list(MODEL.named_parameters())
        NO_DECAY = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in PARAM_OPTIM if not any(nd in n for nd in NO_DECAY)], "weight_decay": 0.01},
            {"params": [p for n, p in PARAM_OPTIM if any(nd in n for nd in NO_DECAY)], "weight_decay": 0.0}
        ]
        OPTIMIZER = AdamW(optimizer_grouped_parameters, lr=self.train_params["learning_rate"])
        TOTAL_STEPS = len(DATASET_LOADERS["train"]) * self.train_params["epochs"]
        SCHEDULER = get_linear_schedule_with_warmup(
            OPTIMIZER,
            num_warmup_steps=0,
            num_training_steps=TOTAL_STEPS
        )
        ## Train
        HISTORY = []
        OVERALL_HISTORY = {
            "train": {},
            "val": {}
        }
        for epoch in tqdm(range(self.train_params["epochs"])):
            ## Train
            TrainData = TrainModel_Epoch(
                MODEL,
                DATASET_LOADERS["train"],     
                OPTIMIZER, 
                SCHEDULER, 
                self.device
            )
            MODEL = TrainData["model"]
            ## Eval
            ValData = self.test(
                DATASETS["Fs"]["val"], DATASETS["Ls"]["val"]
            )
            ## Record
            N_CUREPOCH_HISTORY_TRAIN = TrainData["counter"]
            N_CUREPOCH_HISTORY_VAL = ValData["counter"]
            CUR_HISTORY = {
                "epoch": epoch,
                "train_count": TrainData["counter"],
                "train": {},
                "val": {},
                "train_expanded": TrainData["metrics"]
            }
            for k in TrainData["metrics"][0].keys():
                CUR_HISTORY["train"][k] = np.mean([TrainData["metrics"][i][k] for i in range(N_CUREPOCH_HISTORY_TRAIN)])
            CUR_HISTORY["val"] = ValData["metrics"]
            # for k in ValData["metrics"][0].keys():
                # CUR_HISTORY["val"][k] = np.mean([ValData["metrics"][i][k] for i in range(N_CUREPOCH_HISTORY_VAL)])
            HISTORY.append(CUR_HISTORY)
            ## Print
            print("Epoch: ", epoch)
            print("Train Loss: ", CUR_HISTORY["train"]["loss"])
            print("Train Acc: ", CUR_HISTORY["train"]["acc"])
            print("Val Loss: ", CUR_HISTORY["val"]["loss"])
            print("Val Acc: ", CUR_HISTORY["val"]["acc"])
            ## Save Best Model
            if len(OVERALL_HISTORY["val"].keys()) == 0 or OVERALL_HISTORY["val"]["accuracy"] < CUR_HISTORY["val"]["accuracy"]:
                OVERALL_HISTORY = CUR_HISTORY
                torch.save(MODEL.state_dict(), os.path.join(self.train_params["save_dir"], "best_model.bin"))
        self.time_data["train"] = Time_Record("Model Training", self.time_data["train"])
        # Record
        self.model = {
            "model": MODEL
        }
        self.history = {
            "history": HISTORY,
            "best_model_info": OVERALL_HISTORY
        }
        self.time_data["train"] = Time_Record("", self.time_data["train"], finish=True)

        return self.model
        
    def visualise(self, disable_plots=False):
        '''
        Visualise
        '''
        # Init
        VisData = {
            "figs": {
                "pyplot": {},
                "plotly_chart": {}
            },
            "data": {}
        }
        Plots = {
            "Train Loss": None,
            "Train Accuracy": None,
            "Val Loss": None,
            "Val Accuracy": None
        }
        Data = {}
        # Get Data
        
        # Plot
        if not disable_plots:
            ## Train Loss
            Plots["Train Loss"] = plt.figure()
            plt.plot([self.history["history"][i]["train"]["loss"] for i in range(len(self.history["train"]))])
            plt.title("Train Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            ## Train Accuracy
            Plots["Train Accuracy"] = plt.figure()
            plt.plot([self.history["history"][i]["train"]["accuracy"] for i in range(len(self.history["train"]))])
            plt.title("Train Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            ## Val Loss
            Plots["Val Loss"] = plt.figure()
            plt.plot([self.history["history"][i]["val"]["loss"] for i in range(len(self.history["train"]))])
            plt.title("Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            ## Val Accuracy
            Plots["Val Accuracy"] = plt.figure()
            plt.plot([self.history["history"][i]["val"]["accuracy"] for i in range(len(self.history["train"]))])
            plt.title("Val Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
        ## Record
        Data["Time"] = self.time_data
        ## CleanUp
        for k in Plots.keys():
            if Plots[k] is not None:
                plt.close(Plots[k])
        # Record
        VisData["figs"]["plotly_chart"] = Plots
        VisData["data"] = Data

        return VisData

    def tokenize(self,
        texts,
        **params
        ):
        '''
        Tokenize

        Tokenize the text features.

        Inputs:
            - texts : Text Input (N_Samples, 1)
        Outputs:
            - input_ids : Tokenized Text Input (N_Samples, MAX_LEN)
            - OtherData : Other Data (Attention Masks, etc)
        '''
        # Init
        Fs = np.array(texts)
        MAX_LEN = self.dataset_params["max_len"]
        TOKENIZER = self.tokenizer
        TOKEN_DATA = {
            "input_ids": [],
            "attention_mask": []
        }
        # Encode
        for i in range(Fs.shape[0]):
            F_encoded = TOKENIZER.encode_plus(
                Fs[i][0],
                max_length=MAX_LEN,
                add_special_tokens=True,
                return_token_type_ids=False,
                pad_to_max_length=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ## Input IDs
            input_ids = pad_sequences(
                F_encoded["input_ids"], 
                maxlen=MAX_LEN, dtype=torch.Tensor, truncating="post", padding="post"
            ).astype(dtype="int64")
            input_ids = torch.tensor(input_ids)
            TOKEN_DATA["input_ids"].append(input_ids)
            ## Attention Mask
            attention_mask = pad_sequences(
                F_encoded["attention_mask"], 
                maxlen=MAX_LEN, dtype=torch.Tensor, truncating="post", padding="post"
            ).astype(dtype="int64")
            attention_mask = torch.tensor(attention_mask)
            TOKEN_DATA["attention_mask"].append(attention_mask)
        # Finalize
        TOKEN_DATA["input_ids"] = torch.cat(TOKEN_DATA["input_ids"], dim=0).reshape(-1, MAX_LEN).to(self.device)
        TOKEN_DATA["attention_mask"] = torch.cat(TOKEN_DATA["attention_mask"], dim=0).reshape(-1, MAX_LEN).to(self.device)

        return TOKEN_DATA

    def test(self,
        Fs, Ls,
        **params
        ):
        '''
        Predict

        Test Model on given features and labels.

        Inputs:
            - Fs : Text Input (N_Samples, 1)
            - Ls : True Label Distributions (N_Samples, Label_Dim)
        Outputs:
            - Metrics : Test Metrics
        '''
        # Init
        Ls = np.array(Ls)
        # Predict
        Ls_pred = self.predict(Fs, record_time=False)
        # Metrics
        Ls_indices = np.argmax(Ls, axis=-1)
        Ls_pred_indices = np.argmax(Ls_pred, axis=-1)
        METRICS = Eval_Basic(Ls_indices, Ls_pred_indices)# , self.features_info["target"]["type"]["categories"])

        return METRICS

    def predict(self,
        Fs, 
        record_time=True,
        **params
        ):
        '''
        Predict

        Predict labels from features.

        Inputs:
            - Fs : Text Input (N_Samples, 1)
        Outputs:
            - Ls : Label Distributions (N_Samples, Label_Dim)
        '''
        if record_time: self.time_data["predict"] = Time_Record("Model - Predict")
        # Init
        Fs = np.array(Fs["input"])
        Ls = None
        MODEL = self.model["model"]
        # Preprocess
        TOKEN_DATA = self.tokenize(Fs)
        if record_time: self.time_data["predict"] = Time_Record("Data Preprocess", self.time_data["predict"])
        # Predict
        outputs = MODEL(
            input_ids=TOKEN_DATA["input_ids"], 
            attention_mask=TOKEN_DATA["attention_mask"]
        )
        if record_time: self.time_data["predict"] = Time_Record("Model Prediction", self.time_data["predict"])
        PROB_DIST = F.softmax(outputs.logits, dim=-1).cpu().detach().numpy()
        Ls = PROB_DIST
        if record_time: self.time_data["predict"] = Time_Record("", self.time_data["predict"], finish=True)

        return Ls
    
    def save(self, path):
        '''
        Save
        '''
        # Init
        paths = {
            "class_obj": os.path.join(path, self.save_paths["class_obj"]),
            "model": os.path.join(path, self.save_paths["model"]),
            "tokenizer": os.path.join(path, self.save_paths["tokenizer"]),
        }
        # Save Model
        os.makedirs(os.path.dirname(paths["model"]), exist_ok=True)
        torch.save(self.model["model"], paths["model"])
        # Save Tokenizer
        os.makedirs(paths["tokenizer"], exist_ok=True)
        self.tokenizer.save_pretrained(paths["tokenizer"])
        # Save Class Object (Ignore some objects while saving)
        dict_data = self.__dict__
        ## Ignore
        save_removed_objects = {
            "model": self.model["model"],
            "tokenizer": self.tokenizer,
        }
        dict_data["model"]["model"] = None
        dict_data["tokenizer"] = None
        ## Save
        with open(paths["class_obj"], "wb") as f: pickle.dump(dict_data, f)
        ## Restore
        self.__dict__["model"]["model"] = save_removed_objects["model"]
        self.__dict__["tokenizer"] = save_removed_objects["tokenizer"]

    def load(self, path):
        '''
        Load
        '''
        # Init
        paths = {
            "class_obj": os.path.join(path, self.save_paths["class_obj"]),
            "model": os.path.join(path, self.save_paths["model"]),
            "tokenizer": os.path.join(path, self.save_paths["tokenizer"]),
        }
        # Load Class Object
        with open(paths["class_obj"], "rb") as f: dict_data = pickle.load(f)
        # Load Model
        dict_data["model"]["model"] = torch.load(paths["model"])
        # Load Tokenizer
        dict_data["tokenizer"] = AutoTokenizer.from_pretrained(paths["tokenizer"])
        # Load
        self.__dict__.update(dict_data)

    def load_pretrained(self, model_id):
        '''
        Load Pretrained Model
        '''
        # Load
        self.tokenizer = AutoTokenizer.from_pretrained(model_id).to(self.device)
        self.model["model"] = AutoModelForSequenceClassification.from_pretrained(
            model_id, 
            num_labels=self.n_classes
        ).to(self.device)

# Main Vars
TASK_FUNCS = {
    "HuggingFace": {
        "class": TextProblems_SentimentAnalysis_HuggingFace,
        "params": {
            "n_classes": 2,
            "dataset_params": {
                "test_size": 0.2,
                "val_size": 0.05,
                "random_state": 0,
                "max_len": 16,
                "batch_size": 4
            },
            "train_params": {
                "batch_size": 4,
                "epochs": 3,
                "learning_rate": 3e-5,
                "save_dir": "_models/temp/"
            },
            "model_params": {
                "model_id": "xlnet-base-cased",
                "load_pretrained": True
            },
            "random_state": 0
        }
    }
}