import os
import math
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup

from sklearn.model_selection import KFold

#Garbage collection mechanism included due to large size of transformer objects
import gc
gc.enable()

#Specifying some essential hyperparameters
NUM_FOLDS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 100
EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]

#Here should be listed the path to a pre-trained XLM-RoBERTa model (or any RoBERTa-based model)
ROBERTA_PATH = "cardiffnlp/twitter-xlm-roberta-base"
TOKENIZER_PATH = "cardiffnlp/twitter-xlm-roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Only first 900 examples were read because they were sentences from Wikipedia and annotated by human annotators, 
# whereas the final 100 examples were from the Leichte Sprache dataset and had a manually assigned score.
train_df = pd.read_csv("training_set.csv", index_col=[0]).iloc[:900, :]
test_df = pd.read_csv("validation_set.csv")
submission_df = pd.read_csv("answer-final.csv")


#Here the subsampling of data is done. Set percentage to 1.0 if you want to train on the complete dataset.
SUBSAMPLE_PERCENTAGE = 0.6
SEED = 1111

random.seed(SEED)
x = random.sample(list(range(900)), int(SUBSAMPLE_PERCENTAGE*900))
y = list(set(range(900)) - set(x))

training_df = train_df.iloc[x].copy()
testing_df = train_df.iloc[y].copy()
train_df = training_df
train_df = train_df.reset_index()


#Load the previously specified tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


#Class for processing the text complexity challenge dataset. 
#MOS is the mean opinion score column in the dataset while Sentence is the column with text.
class ComplexityDataset(Dataset):
    def __init__(self, df, inference=False):
        super().__init__()

        self.df = df        
        self.inference = inference
        self.text = df.Sentence.tolist()
        
        if not self.inference:
            self.target = torch.tensor(df.MOS.values, dtype=torch.float32)        
    
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
 

    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.inference:
            return (input_ids, attention_mask)            
        else:
            target = self.target[index]
            return (input_ids, attention_mask, target)


#Here the model for regression using XLM-RoBERTa is built.
#Two additional layers are added at the end, one for learning the weights of inputs and the other 
# for the final MOS output prediction.
class ComplexityModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(ROBERTA_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        #This is the original XLM-RoBERTa with all 12 encoders.
        self.roberta = AutoModel.from_pretrained(ROBERTA_PATH, config=config)  
            
        #Additional layer that learns the weights (scaled with softmax) to assign to each of the 768 vectors values 
        # from the final layer - similar to the attention meachnism. This way the final layer gets a useful representation.
        self.weighting = nn.Sequential(            
            nn.Linear(768, 512),            
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )        

        #Final linear layer for regreesion, learns to convert 
        self.regressor = nn.Sequential(                        
            nn.Linear(768, 1)                        
        )
        

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)        

        last_layer_hidden = roberta_output.hidden_states[-1]

        weights = self.weighting(last_layer_hidden)
        
        #Multiplying the learned weights with the last hidden layer output produces a useful representation.
        context_vector = torch.sum(weights * last_layer_hidden, dim=1)   
        
        #Final score will be a float value telling us the predicted mean opinion score (MOS)
        return self.regressor(context_vector)


#Sets the random seet for training
def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True


#Calculates the root mean squared error of the model on specified dataset
def eval_mse(model, data_loader):
    model.eval()            
    mse_sum = 0

    with torch.no_grad():
        for batch_num, (input_ids, attention_mask, target) in enumerate(data_loader):                
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)                        
            target = target.to(DEVICE)           
            
            pred = model(input_ids, attention_mask,)                       

            mse_sum += nn.MSELoss(reduction="sum")(pred.flatten(), target).item()       

    return mse_sum / len(data_loader.dataset)


#Predicts the mean opinion scores using the specified model and dataset
def predict(model, data_loader):
    model.eval()

    result = np.zeros(len(data_loader.dataset))    
    index = 0
    
    with torch.no_grad():
        for batch_num, (input_ids, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
                        
            pred = model(input_ids, attention_mask)                        

            result[index : index + pred.shape[0]] = pred.flatten().to("cpu")
            index += pred.shape[0]

    return result

#Trains the specified model for the specified number of epochs on the train loader, evaluates on validation loader
def train(model, model_path, train_loader, val_loader,
          optimizer, scheduler=None, num_epochs=NUM_EPOCHS):    
    best_val_rmse = None
    best_epoch = 0
    step = 0
    last_eval_step = 0
    eval_period = EVAL_SCHEDULE[0][1]    

    start = time.time()

    for epoch in range(num_epochs):                           
        val_rmse = None         

        for batch_num, (input_ids, attention_mask, target) in enumerate(train_loader):              
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)            
            target = target.to(DEVICE)                        

            optimizer.zero_grad()
            
            model.train()

            pred = model(input_ids, attention_mask)
                                                        
            mse = nn.MSELoss(reduction="mean")(pred.flatten(), target)
                        
            mse.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()
            
            if step >= last_eval_step + eval_period:
                # Evaluate the model on validation loader
                elapsed_seconds = time.time() - start
                num_steps = step - last_eval_step
                print(f"\n{num_steps} steps took {elapsed_seconds:0.3} seconds")
                last_eval_step = step
                
                val_rmse = math.sqrt(eval_mse(model, val_loader))                            

                print(f"Epoch: {epoch} batch_num: {batch_num}", 
                      f"val_rmse: {val_rmse:0.4}")

                for rmse, period in EVAL_SCHEDULE:
                    if val_rmse >= rmse:
                        eval_period = period
                        break                               
                
                if not best_val_rmse or val_rmse < best_val_rmse:                    
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_path)
                    print(f"New best value of RMSE: {best_val_rmse:0.4}")
                else:       
                    print(f"Still best value of RMSE: {best_val_rmse:0.4}",
                          f"(from epoch {best_epoch})")                                    
                    
                start = time.time()
                                            
            step += 1
                        
    
    return best_val_rmse

#Creates the optimizer
def create_optimizer(model):
    named_parameters = list(model.named_parameters())    
    
    roberta_parameters = named_parameters[:197]    
    attention_parameters = named_parameters[199:203]
    regressor_parameters = named_parameters[203:]
        
    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    parameters.append({"params": attention_group})
    parameters.append({"params": regressor_group})

    #Learning rate changes dynamically during training. The idea behind this is that lower layers learn to represent 
    # the morphological and syntactical features, while higher layers learn to represent the semantical meaning. 
    # Since the low-level features are more important for this task, they are trained more thoroughly.
    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01

        lr = 2e-5

        if layer_num >= 69:        
            lr = 5e-5

        if layer_num >= 133:
            lr = 1e-4

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return AdamW(parameters)


#Activate the garbage collection
gc.collect()

SEED = 1000
list_val_rmse = []
kfold = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)

#Start the training procedure
for fold, (train_indices, val_indices) in enumerate(kfold.split(train_df)):    
    print(f"\nFold {fold + 1}/{NUM_FOLDS}")
    model_path = f"model_{fold + 1}.pth"
        
    set_random_seed(SEED + fold)
    
    train_dataset = ComplexityDataset(train_df.loc[train_indices])    
    val_dataset = ComplexityDataset(train_df.loc[val_indices])    
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              drop_last=True, shuffle=True, num_workers=2)    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            drop_last=False, shuffle=False, num_workers=2)    
        
    set_random_seed(SEED + fold)    
    
    model = ComplexityModel().to(DEVICE)
    
    optimizer = create_optimizer(model)                        
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=NUM_EPOCHS * len(train_loader),
        num_warmup_steps=50)    
    
    list_val_rmse.append(train(model, model_path, train_loader,
                               val_loader, optimizer, scheduler=scheduler))

    del model
    gc.collect()
    
    print("\nPerformance estimates:")
    print(list_val_rmse)
    print("Mean:", np.array(list_val_rmse).mean())

gc.collect()
test_df = pd.read_csv("part2_public.csv")

submission_df = pd.read_csv("answer-final.csv", index_col=[0])
all_predictions = np.zeros((len(list_val_rmse), len(test_df)))

test_dataset = ComplexityDataset(test_df, inference_only=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         drop_last=False, shuffle=False, num_workers=2)

#Since 5 models were trained in total, this takes the predictions of all 5 of them and later they are averaged
for index in range(len(list_val_rmse)):            
    model_path = f"model_{index + 1}.pth"
    print(f"\nUsing {model_path}")
                        
    model = ComplexityModel()
    model.load_state_dict(torch.load(model_path))    
    model.to(DEVICE)
    
    all_predictions[index] = predict(model, test_loader)
    
    del model
    gc.collect()


#Generate final predictions and the submission file
predictions = all_predictions.mean(axis=0)
submission_df.MOS = predictions
print(submission_df)
submission_df.to_csv("answer.csv", index=True)


