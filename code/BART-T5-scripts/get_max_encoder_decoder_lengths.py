#this script helps you figure out the max encoder and decoder lengths to use for T5 and BART for both training and inference by seeing the percentage of examples which are above different token length thresholds (e.g. 32) and the max token length of all examples
#in the end we chose max_encoder_len = 32 and max_decoder_len = 128 for BART, and 35 and 128 for T5

#the two commented-out lines below contain required packages/libraries that must be installed:
#pip install datasets
#pip install --no-cache-dir transformers==4.3.3 sentencepiece==0.1.95


train_json_filename = 'data/HF_training_data/combined/combined_train_HF.json' #filename containing example texts to get the max tokenizer lengths of
bart_or_t5 = "t5" #which model, bart or t5


import json

# load data from json file. format: list of dictionaries, each containing one example
def load_from_json(filename):
    with open(filename) as f:
        loaded_json = json.loads(f.read())
    print(type(loaded_json))
    print(loaded_json[0])
    inputs = [x['input'] for x in loaded_json]
    outputs = [x['output'] for x in loaded_json]
    print('input:', inputs[0])
    print('output:', outputs[0])
    return loaded_json

loaded_train_json = load_from_json(train_json_filename)


#convert loaded json format to format required to load into huggingface dataset. edited_key argument same as earlier
def convert_json_format(loaded_json):#(loaded_train_json,loaded_val_json):
    loaded_dict = {}
    input_lst = [x['input'] for x in loaded_json]
    print(len(input_lst))
    output_lst = [x['output'] for x in loaded_json]
    print(len(output_lst))
    loaded_dict['input'] = input_lst
    loaded_dict['output'] = output_lst
    print(loaded_dict['input'][:3])
    print(loaded_dict['output'][:3])
    return loaded_dict
 
converted_train_json = convert_json_format(loaded_train_json)


from datasets import Dataset

#convert the converted_json to huggingface dataset format
def load_huggingface_dataset_from_dict(converted_json):
    #train_dataset = Dataset.from_dict(converted_json["train"])
    #val_dataset = Dataset.from_dict(converted_json["validation"])
    loaded_hf_dataset = Dataset.from_dict(converted_json)
    print(type(loaded_hf_dataset))
    print(loaded_hf_dataset)
    return loaded_hf_dataset

commongen_train = load_huggingface_dataset_from_dict(converted_train_json)
print(commongen_train[:5])


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM                   
from transformers import Trainer, TrainingArguments

# see https://discuss.huggingface.co/t/error-with-new-tokenizers-urgent/2847/4
if bart_or_t5 == "bart":
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", use_fast=False)
if bart_or_t5 == "t5":
    tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)
    

#below code tests that the tokenizer is loaded correctly
#test0 = tokenizer.tokenize(loaded_test_json[0]["input"])
test1 = tokenizer.tokenize("hello <s>hello test</s>")
test2 = tokenizer.tokenize("hello <sep> bye")
test3 = tokenizer.tokenize("hello _eos bye")
test4 = tokenizer.tokenize("hello \n bye")
test5 = tokenizer.tokenize("hello {bye}")
#print(test0)
print(test1)
print(test2)
print(test3)
print(test4)
print(test5)

#test0_n = tokenizer.convert_tokens_to_string(test0)
test1_n = tokenizer.convert_tokens_to_string(test1)
test2_n = tokenizer.convert_tokens_to_string(test2)
test3_n = tokenizer.convert_tokens_to_string(test3)
test4_n = tokenizer.convert_tokens_to_string(test4)
test5_n = tokenizer.convert_tokens_to_string(test5)

#print(test0_n)
print(test1_n)
print(test2_n)
print(test3_n)
print(test4_n)
print(test5_n)

print(tokenizer.eos_token)
print(tokenizer.eos_token_id)
print(tokenizer.decode([1]))
print(tokenizer.sep_token)
print(tokenizer.sep_token_id)



#determine lengths

def map_to_length(x):
    x["article_len"] = len(tokenizer(x["input"]).input_ids)
    x["article_longer_32"] = int(x["article_len"] > 32)
    x["article_longer_64"] = int(x["article_len"] > 64)
    x["article_longer_128"] = int(x["article_len"] > 128)
    x["article_longer_256"] = int(x["article_len"] > 256)
    x["article_longer_512"] = int(x["article_len"] > 512)
    x["article_longer_1024"] = int(x["article_len"] > 1024)
    x["summary_len"] = len(tokenizer(x["output"]).input_ids)
    x["summary_longer_32"] = int(x["summary_len"] > 32)
    x["summary_longer_64"] = int(x["summary_len"] > 64)
    x["summary_longer_128"] = int(x["summary_len"] > 128)
    x["summary_longer_256"] = int(x["summary_len"] > 256)
    return x

#compute and print out tokenizer and decoder length stats (in terms of percentage of texts above different length thresholds)

def compute_and_print_stats(x):
  if len(x["article_len"]) == sample_size:
    print("sample_size: ", sample_size)
    print(
        "Article Mean: {}, %-Articles > 32: {}, %-Articles > 64: {}, %-Articles > 128: {}, %-Articles > 256: {}, %-Articles > 512: {}, %-Articles > 1024: {}, Summary Mean: {}, %-Summary > 32: {}, %-Summary > 64: {}, %-Summary > 128: {}, %-Summary > 256: {}".format(
            sum(x["article_len"]) / sample_size,
            sum(x["article_longer_32"]) / sample_size, 
            sum(x["article_longer_64"]) / sample_size, 
            sum(x["article_longer_128"]) / sample_size, 
            sum(x["article_longer_256"]) / sample_size, 
            sum(x["article_longer_512"]) / sample_size, 
            sum(x["article_longer_1024"]) / sample_size, 
            sum(x["summary_len"]) / sample_size,
            sum(x["summary_longer_32"]) / sample_size,
            sum(x["summary_longer_64"]) / sample_size,
            sum(x["summary_longer_128"]) / sample_size,
            sum(x["summary_longer_256"]) / sample_size
        )
    )


sample_size = len(commongen_train)
train_data_stats = commongen_train.select(range(len(commongen_train))).map(map_to_length, num_proc=1)

output = train_data_stats.map(
  compute_and_print_stats, 
  batched=True,
  batch_size=-1,
)


#compute and print out max token lengths of inputs and outputs

train_article_lens = []
train_summary_lens = []
for x in tqdm(train_data_stats):
    train_article_lens.append(x["article_len"])
    train_summary_lens.append(x["summary_len"])
print(max(train_article_lens))
print(max(train_summary_lens))