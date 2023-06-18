import os
import sys

#CHANGE values in this cell appropriately
size = sys.argv[1] #CHANGE: IMPORTANT: either 'base' or 'large', base model is smaller and quicker to train and may be sufficient for your purposes. especially since T5-large is much much slower and compute heavy thane T5-base due to having to disable fp16
lr = float(sys.argv[2]) #CHANGE: IMPORTANT: need to choose a good learning rate. some reasonable values to try are 5e-06, 1e-05, 2e-05, 3e-05, 5e-05
default_seed=int(sys.argv[3]) #CHANGE: IMPORTANT: seed1: 42, seed2: 24 (typically you train two seeds/versions of a model and take average, for your purposes probably enough to stick with one seed)
ENCODER_MAX_LENGTH = int(sys.argv[4]) #CHANGE: IMPORTANT: set this to reasonable value to ensure most input texts in your dataset can fit onto the encoder model. this is the number of tokens, but you can approximate it with the number of words in the input texts. reasonable values: 32, 64, 128 (powers of 2). 64 should be enough for most purposes...
DECODER_MAX_LENGTH = int(sys.argv[5]) #CHANGE: IMPORTANT: set this to reasonable value to ensure most output texts in your dataset can fit onto the decoder model. this is the number of tokens, but you can approximate it with the number of words in the output texts. reasonable values: 32, 64, 128 (powers of 2). 64 should be enough for most purposes...
model_type = sys.argv[6] #CHANGE: a string to describe the type of trained model, e.g. 'baseline' vs. 'top1'

out_path = f"trained_T5_models/T5-{size}_{default_seed}_{model_type}_{lr}" #IMPORTANT: path to save trained checkpoints to (each in their individual folder)
train_json_filename = sys.argv[7] #IMPORTANT: json file containing the training examples. will send some example files so you know the format (should be list of dictionaries, each corresponding to one example with keys of 'input' and 'output' containing the input and output text for that example, respectively)
val_json_filename = sys.argv[8] #IMPORTANT: json file containing the validation examples. will send some example files so you know the format (should be list of dictionaries, each corresponding to one example with keys of 'input' and 'output' containing the input and output text for that example, respectively)

"""# Preliminary Things to Run"""

BATCH_SIZE = int(sys.argv[9])
num_epochs = int(sys.argv[10])

if size == 'base':
    #num_epochs = 20 #number of epochs to train for. typically i recommend 20 for base and 10 for large.
    #BATCH_SIZE=64 #reduce this by a power of 2 (e.g. to 32) if running into "CUDA out of memory" errors during training
    default_warmup_steps = 400
    default_fp16=True #this makes training much faster. however, can only be used on T5-base (not T5-large)
elif size == 'large':
    #num_epochs = 10 #number of epochs to train for. typically i recommend 20 for base and 10 for large.
    #BATCH_SIZE=8 #reduce this by a power of 2 (e.g. to 4) if running into "CUDA out of memory" errors during training
    default_warmup_steps = 1200
    default_fp16=True #must be disabled for T5-large. hence, training T5-large is much slower and compute heavy
print("BATCH_SIZE: ", BATCH_SIZE)

input_model = f"t5-{size}"


import torch
from packaging import version
if version.parse(torch.__version__) < version.parse("1.6"):
    from .file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast
    
GPU = sys.argv[11]
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
#device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
#print('Using device:', device)


from datasets import load_dataset, load_metric                                               
#from torch.cuda.amp import autocast # need torch >=1.6                          
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM      
except:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM                   
from transformers import Trainer, TrainingArguments
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer

import transformers
print(transformers.__version__)

t5_tokenizer = T5Tokenizer.from_pretrained(input_model, use_fast=False)

"""# Load CommonGen Data into HuggingFace Dataset"""

import json

# load data from json file. format: list of dictionaries, each containing one example, where keys of each dict are "document", "edited_key" (specified as argument), "id", and "summary" (and possibly additional keys, these are just the mandatory ones)
def load_from_json(filename):
    with open(filename) as f:
        loaded_json = json.loads(f.read())
    #print(type(loaded_json))
    print(loaded_json[0])
    inputs = [x['input'] for x in loaded_json]
    outputs = [x['output'] for x in loaded_json]
    #print('input:', inputs[0])
    #print('output:', outputs[0])
    return loaded_json

loaded_train_json = load_from_json(train_json_filename)
loaded_val_json = load_from_json(val_json_filename)

#modified : this cell has been inserted

#convert loaded json format to format required to load into huggingface dataset. edited_key argument same as earlier
def convert_json_format(loaded_json):#(loaded_train_json,loaded_val_json):
    loaded_dict = {}
    input_lst = [x['input'] for x in loaded_json]
    print(len(input_lst))
    output_lst = [x['output'] for x in loaded_json]
    print(len(output_lst))
    loaded_dict['input'] = input_lst
    loaded_dict['output'] = output_lst
    #print(loaded_dict['input'][0])
    #print(loaded_dict['output'][0])
    return loaded_dict

converted_train_json = convert_json_format(loaded_train_json)
converted_val_json = convert_json_format(loaded_val_json)

#modified : this cell has been inserted

from datasets import Dataset

#convert the converted_json to huggingface dataset format
def load_huggingface_dataset_from_dict(converted_json):
    #train_dataset = Dataset.from_dict(converted_json["train"])
    #val_dataset = Dataset.from_dict(converted_json["validation"])
    loaded_hf_dataset = Dataset.from_dict(converted_json)
    #print(type(loaded_hf_dataset))
    #print(loaded_hf_dataset)
    return loaded_hf_dataset

commongen_train = load_huggingface_dataset_from_dict(converted_train_json)
commongen_val = load_huggingface_dataset_from_dict(converted_val_json)

print(commongen_train[0])
print(commongen_val[0])

"""#Main Training Code"""

LANGUAGE = 'en'

BEAM_SIZE = 5 #CHANGED : was 2
DECODER_EARLY_STOPPING = True #CHANGED : added #Vestigial for Pegasus
DECODER_LENGTH_PENALTY = 0.6 #CHANGED : added
DECODER_MIN_LENGTH = 1 #CHANGED : added
NO_REPEAT_NGRAM_SIZE = 3
device = 'cuda'

import nltk; nltk.download('punkt')
t5_model = T5ForConditionalGeneration.from_pretrained(input_model)

#NOTE: this fixes a bug in v. 4.1.1 where the mBART model checkpoints are not saved fully. It does not affect 4.2.
if version.parse(transformers.__version__) <= version.parse("4.1.1"):
    t5_model._keys_to_ignore_on_load_missing = None
    t5_model._keys_to_ignore_on_save = None
t5_model.to(device)

import json
def batch_t5_tokenize(dataset_batch, tokenizer, decoder_max_length=DECODER_MAX_LENGTH):    
    # concatenate the concept names for each example in the batch               
    input_text  = dataset_batch["input"]                                         
    output_text = dataset_batch["output"]                                      
    res = tokenizer.prepare_seq2seq_batch(src_texts=input_text,
                                          tgt_texts=output_text,
                                          src_lang=LANGUAGE,
                                          tgt_lang=LANGUAGE,
                                          max_length=ENCODER_MAX_LENGTH,
                                          max_target_length=decoder_max_length,
                                          padding="max_length",truncation=True)
    return res

#around 9 minutes to run on entire xsum training split (V100)
train_tokenized = commongen_train.map(                                        
    lambda batch: batch_t5_tokenize(batch, t5_tokenizer),                  
    batched=True,load_from_cache_file=False)
#print(train_tokenized[:10])

valid_tokenized = commongen_val.map(                                   
    lambda batch: batch_t5_tokenize(batch, t5_tokenizer),                  
    batched=True,load_from_cache_file=False)
#print(valid_tokenized[:10])

class MySeq2SeqTrainerT5(Trainer):
    def __init__(
        self,
        num_beams=5, max_length=32, min_length=1, length_penalty=0.6, early_stopping=True,no_repeat_ngram_size = 3, #prefix = "summarize: ",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_beams = num_beams
        self.max_length = max_length
        self.min_length = min_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.no_repeat_ngram_size = no_repeat_ngram_size
        #self.prefix = prefix
        self.lang_id = self.tokenizer.encode(LANGUAGE)[0]
    # tells the trainer to use the generate funtion to predict full sentences at test time
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        #if ignore_keys is None:
        #    if hasattr(self.model, "config"):
        #        ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
        #    else:
        #        ignore_keys = []
        # compute loss with labels first
        with torch.no_grad():
            if self.args.fp16 and _use_native_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                loss = outputs[0].mean().detach()
            else:
                loss = None
        # if we're only computing the conditional log-likelihood, return
        if prediction_loss_only:
            return (loss, None, None)
        # otherwise run model.generate() to get predictions
        if isinstance(model, torch.nn.DataParallel):
            preds = model.module.generate(
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                length_penalty = self.length_penalty,
                num_beams=self.num_beams,
                min_length = self.min_length,
                max_length=self.max_length,
                early_stopping = self.early_stopping,
                no_repeat_ngram_size = self.no_repeat_ngram_size,
                decoder_start_token_id = 0
            )
        else:
            preds = model.generate(
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                length_penalty = self.length_penalty,
                num_beams=self.num_beams,
                min_length = self.min_length,
                max_length=self.max_length,
                early_stopping = self.early_stopping,
                no_repeat_ngram_size = self.no_repeat_ngram_size,
                decoder_start_token_id = 0
            )
        if len(preds) == 1:
            preds = preds[0]
        # pad predictions if necessary so they can be concatenated across batches
        if preds.shape[-1] < self.max_length:
            preds = torch.nn.functional.pad(
                preds, (0, self.max_length-preds.shape[-1]),
                mode='constant',
                value=self.tokenizer.pad_token_id
            )
        # post-process labels
        if has_labels:
            labels = inputs.get('labels')
        else:
            labels = None
        return (loss, preds, labels)

rouge_scorer = load_metric("rouge")

def compute_rouge_metrics_t5(pred):                                                
    labels_ids = pred.label_ids                                                 
    pred_ids = pred.predictions                                                 
    # all unnecessary tokens are removed
    pred_str = t5_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)  
    labels_ids[labels_ids == -100] = t5_tokenizer.pad_token_id                
    label_str = t5_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    # compute the metric proper                                                 
    rouge_results = rouge_scorer.compute(                                       
        predictions=pred_str,                                                   
        references=label_str,                                                   
        rouge_types=["rouge1", "rouge2", "rougeL"],                                       
        use_agregator=True, use_stemmer=False,                                  
    )                                                                           
    return {    
        "rouge1_fmeasure": round(rouge_results['rouge1'].mid.fmeasure, 4),         
        "rouge2_fmeasure": round(rouge_results['rouge2'].mid.fmeasure, 4),      
        "rougeL_fmeasure": round(rouge_results['rougeL'].mid.fmeasure, 4),      
    }

#NOTE: this will break if BATCH_SIZE=1
LEARNING_RATE=lr
GRADIENT_ACCUMULATION_STEPS=1

t5_train_args = TrainingArguments(                                            
    output_dir=out_path,                                        
    do_train=True,                                                              
    do_eval=True,                                                               
    evaluation_strategy="epoch",                                                
    logging_steps=50,                                                          
    # optimization args, the trainer uses the Adam optimizer                    
    # and has a linear warmup for the learning rate                             
    per_device_train_batch_size=BATCH_SIZE,                                             
    per_device_eval_batch_size=BATCH_SIZE,                                              
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,                                              
    learning_rate=LEARNING_RATE,                                                        
    num_train_epochs=num_epochs,                                                         
    warmup_steps=default_warmup_steps,                                                        
    # misc args
    fp16_opt_level='O1',
    fp16=default_fp16,
    adam_epsilon=1e-08,
    seed=default_seed,                                                           
    disable_tqdm=False,                                                         
    load_best_model_at_end=True,                                                
    metric_for_best_model="rouge2_fmeasure",
    save_strategy="epoch",
    save_total_limit=1    
)

t5_trainer = MySeq2SeqTrainerT5(                                                
    num_beams=BEAM_SIZE, max_length=DECODER_MAX_LENGTH, min_length=DECODER_MIN_LENGTH, length_penalty=DECODER_LENGTH_PENALTY, early_stopping=DECODER_EARLY_STOPPING,no_repeat_ngram_size = NO_REPEAT_NGRAM_SIZE,
    model=t5_model,                                                           
    args=t5_train_args,                                                       
    train_dataset=train_tokenized,                                              
    eval_dataset=valid_tokenized,                                               
    tokenizer=t5_tokenizer,                                                   
    compute_metrics=compute_rouge_metrics_t5,                                      
)

import time
start = time.time()

if input_model != "t5-base" and input_model != "t5-large":
    t5_trainer.train(resume_from_checkpoint=input_model)
else:
    t5_trainer.train()

end = time.time()
print("time taken (seconds): ", end-start)