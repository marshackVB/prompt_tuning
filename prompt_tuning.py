# Databricks notebook source
# MAGIC %md ## Notes
# MAGIC
# MAGIC
# MAGIC
# MAGIC  - [Model sizes](https://huggingface.co/transformers/v2.9.1/pretrained_models.html). 
# MAGIC  - https://discuss.huggingface.co/t/labels-in-language-modeling-which-tokens-to-set-to-100/2346. 
# MAGIC  - https://discuss.huggingface.co/t/how-to-structure-labels-for-token-classification/1216
# MAGIC  - https://huggingface.co/docs/transformers/model_doc/flava#transformers.FlavaConfig.ce_ignore_index
# MAGIC  

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          default_data_collator, 
                          get_linear_schedule_with_warmup,
                          TrainingArguments,
                          Trainer,
                          pipeline,
                          EarlyStoppingCallback)

from peft import (get_peft_config, 
                  get_peft_model, 
                  PromptTuningInit, 
                  PromptTuningConfig, 
                  TaskType, 
                  PeftType, 
                  PeftConfig, 
                  PeftModel)
                  
import torch
from datasets import load_dataset, Dataset, DatasetDict
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessor import PromptTuneTokenize

# COMMAND ----------

# MAGIC %md Config

# COMMAND ----------

tokenizer.pretrained_vocab_files_map['vocab_file']['bert-base-uncased']

# COMMAND ----------

{"bos_token": "<s>", 
 "eos_token": "</s>", 
 "unk_token": "<unk>", 
 "pad_token": "<pad>"}

# COMMAND ----------

device = "cuda"
model_name_or_path = "bigscience/bloomz-560m"
text_column = "Tweet text"
label_column = "text_label"
max_length = 64
dataset_name = "twitter_complaints"

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# COMMAND ----------

tokenizer.all_special_ids

# COMMAND ----------

tokenizer.all_special_tokens

# COMMAND ----------

min(tokenizer.vocab.values())

# COMMAND ----------

tokenizer.decode([3])

# COMMAND ----------

dir(tokenizer)

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = load_dataset("ought/raft", dataset_name)

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]

dataset = dataset.map(lambda x: {"text_label": [classes[label] for label in x["Label"]]},
                      batched=True,
                      num_proc=1)

# COMMAND ----------

preprocessor = PromptTuneTokenize(dataset = dataset, 
                                  text_column = "Tweet text", 
                                  label_column = "text_label", 
                                  max_length = 64, 
                                  tokenizer = model_name_or_path)

cols_to_remove = dataset["train"].column_names
processed_datasets = preprocessor.process_tokens(cols_to_remove)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["train"]

"""
batch_size = 32
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
"""

# COMMAND ----------

train_dataset

# COMMAND ----------

# MAGIC %md Trainer

# COMMAND ----------

peft_config = PromptTuningConfig(task_type = TaskType.CAUSAL_LM,
                                 prompt_tuning_init=PromptTuningInit.TEXT,
                                 num_virtual_tokens=8,
                                 prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
                                 tokenizer_name_or_path = model_name_or_path
                                )

def init_model(peft_config=peft_config):
  device = "cuda"
  #load_in_8bit=True
  model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", load_in_8bit=True)
  model = get_peft_model(model, peft_config)
  return model.to(device)

model = init_model()

training_args = {"output_dir": "/checkpoints",
                 "overwrite_output_dir": True,
                 "per_device_train_batch_size": 8,
                 "per_device_eval_batch_size": 8,
                 "weight_decay": 0.01,
                 "num_train_epochs": 5,
                 "save_strategy": "epoch", 
                 "evaluation_strategy": "epoch",
                 "logging_strategy": "epoch",
                 #"load_best_model_at_end": True, # Throughs an error during 8-bit training
                 "save_total_limit": 2,
                 "metric_for_best_model": "eval_loss", #https://discuss.huggingface.co/t/early-stopping-training-using-validation-loss-as-the-metric-for-best-model/31378
                 "greater_is_better": False,
                 "seed": 123,
                 "report_to": "none",
                 "gradient_accumulation_steps": 1,
                 #"fp16": True,
                 "learning_rate": 3e-2}

#early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.005)
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=1.0)
                   
trainer = Trainer(model = model,
                  args = TrainingArguments(**training_args),
                  train_dataset = train_dataset, 
                  eval_dataset = eval_dataset)
                  #callbacks=[early_stopping_callback])

trainer.train()

# Full precision: 56.70 seconds
# fp16 precision: 23.26 seconds
# 8bit precision: 16.82 seconds

# COMMAND ----------

# MAGIC %md Save model

# COMMAND ----------

# MAGIC %sh mkdir /peft_config

# COMMAND ----------

model.save_pretrained('/peft_config')

# COMMAND ----------

# MAGIC %sh ls /peft_config

# COMMAND ----------

# MAGIC %md Load model

# COMMAND ----------

 'bigbird_pegasus', 'blip_2', 'bloom', 'bridgetower', 'codegen', 'deit', 'esm', 
    'gpt2', 'gpt_bigcode', 'gpt_neo', 'gpt_neox', 'gpt_neox_japanese', 'gptj', 'gptsan_japanese', 
    'lilt', 'llama', 'longformer', 'longt5', 'luke', 'm2m_100', 'mbart', 'mega', 'mt5', 'nllb_moe', 
    'open_llama', 'opt', 'owlvit', 'plbart', 'roberta', 'roberta_prelayernorm', 'rwkv', 'switch_transformers', 
    't5', 'vilt', 'vit', 'vit_hybrid', 'whisper', 'xglm', 'xlm_roberta'

# COMMAND ----------

from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
   load_in_4bit=True
   #load_in_8bit=True
)

config = PeftConfig.from_pretrained('/peft_config')
#quantization_config=double_quant_config
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
model = PeftModel.from_pretrained(model, '/peft_config')

model.to(device)

# COMMAND ----------

inputs = tokenizer(
    f'{text_column} : {"@nationalgridus I have no water and the bill is current and paid. Can you do something about this?"} Label : ',
    return_tensors="pt",
)

with torch.no_grad():
  inputs = {k: v.to(device) for k, v in inputs.items()}
  outputs = model.generate(
      input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
  )

print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Old Code

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------





def init_model(peft_config):
  device = "cuda"
  model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map_"auto", load_in_8bit=True)
  model = get_peft_model(model, peft_config)
  return model.to(device)

model = init_model(peft_config)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# COMMAND ----------

model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

# COMMAND ----------

# MAGIC %sh mkdir /peft_config

# COMMAND ----------

model.save_pretrained('/peft_config')

# COMMAND ----------

config = PeftConfig.from_pretrained('/peft_config')
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, '/peft_config')

model.to(device)

# COMMAND ----------

results = []

for row in dataset["train"]:
  inputs = tokenizer(
    f'{text_column} : {row[text_column]} Label : ',
    return_tensors="pt",
)
  
  with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
    )
    
    prediction = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    prediction = prediction[0].split(":")[-1].strip()
    results.append((prediction, row['text_label']))

# COMMAND ----------

len(results)
correct_results = [True if pred == actual else False for pred, actual in results]

# COMMAND ----------

correct = 0
for i in correct_results:
  if i is True:
    correct +=1

correct / len(results)

# COMMAND ----------

from transformers import pipeline
generator = pipeline(model="bigscience/bloom-560m", max_length=225)

# COMMAND ----------

dataset["train"][3]

# COMMAND ----------

result = generator(
  """Classify if the tweet is a complaint or not. Below are some examples. Return only the label
  
  Tweet:@HMRCcustomers No this is my first job.

  Label: no_complain

  Tweet: If I can't get my 3rd pair of @beatsbydre powerbeats to work today I'm doneski man. This is a slap in my balls. Your next @Bose @BoseService.

  Label: complaint

  Tweet: @EE On Rosneath Arial having good upload and download speeds but terrible latency 200ms. Why is this.

  Label:
  """)

print(result[0]['generated_text'])

# COMMAND ----------



# COMMAND ----------

inputs = tokenizer(
    f'{text_column} : {"@nationalgridus I have no water and the bill is current and paid. Can you do something about this?"} Label : ',
    return_tensors="pt",
)

model.to(device)

with torch.no_grad():
  inputs = {k: v.to(device) for k, v in inputs.items()}
  outputs = model.generate(
      input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
  )
  
  #print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
  prediction = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

# COMMAND ----------

prediction[0].split(":")[-1].strip()

# COMMAND ----------

import re
regex_pattern = "\bcomplaint\b"

match = re.search(regex_pattern, prediction[0])
if match:
    extracted_text = match.group(0)
    print(extracted_text)  # Output: complaint

# COMMAND ----------

dataset["train"][0]

# COMMAND ----------

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
classes

# COMMAND ----------

dataset = dataset.map(lambda x: {"text_label": [classes[label] for label in x["Label"]]},
                      batched=True,
                      num_proc=1)

# COMMAND ----------

dataset["train"][0]

# COMMAND ----------



# COMMAND ----------

tokenizer.eos_token_id

# COMMAND ----------

classes[0]

# COMMAND ----------

tokenizer(classes[0])

# COMMAND ----------

dataset['train'][text_column][0]

# COMMAND ----------

text_column

# COMMAND ----------

class PromptTuneTokenize:
    def __init__(self, dataset, text_column, label_column, max_length, tokenizer):
        self.dataset = dataset
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def pad_tokens(self, model_inputs, labels, batch_size):
        """
        Iterate through every observation in the batch of tokenized texts and labels. Add a 
        padding token to the end of each observation's label ids. Then, combine the tokenized
        text and label ids into a single sequence. 

        Extend the length of the label array such that its size is equal to the combined text and label
        ids array. To extend the array, prepend the sequence [-100] * len(text_ids) to the label array. This
        makes it possible to distinguish beteween text tokens and label tokens.

        Lastly, extend the attention mask to the length of the combined text and label ids.

        Example for a single observation - all arrays or of equal length:

        Combine the input text and label tokens (padding token is append to the end of the label tokens):
        [227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 77658, 915, 210, 1936, 106863, 3]

        Extend the label array (the label ids are [1936, 106863]):
        [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1936, 106863, 3]

        Extend the attention mask:
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        """
        for i in range(batch_size):

            sample_input_ids = model_inputs["input_ids"][i]

            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]

            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids

            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids

            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

            return (model_inputs, labels)
        


    def tokens_to_tensors(self, model_inputs, labels, batch_size):
        """
        Iterate through every observation in the batch of tokenized texts and labels. Extend the lengths of the
        input_ids, attention_mask, and labels arrays to the max_length. For input_ids, prepend the sequence using
        the tokenizer's pad_token_id. For attention_mask, prepend the sequence using 0. For labels, prepend using
        the value [-100].Finally, convert the arrays to tensors, recoming any tokens ids that exceed the max_length.
        

        Example for a single observation - all arrays are of length max_length:

        input_ids:
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
         3, 3, 3, 3, 3, 3, 3, 3, 227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 
         77658, 915, 210, 1936, 106863, 3]

        attention_mask:
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
   
        labels:
        [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, 1936, 106863, 3]
        """

        for i in range(batch_size):

            sample_input_ids = model_inputs["input_ids"][i]

            label_input_ids = labels["input_ids"][i]

            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids

            model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]

            labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids

            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])

            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])

            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])

            return (model_inputs, labels)


    def preprocess(self, examples):
        """
        Apply processing to the tokenized input text and labels, given a batch of examples. This
        function is intended to be mapped to a dataset.
        """
        batch_size = len(examples[self.text_column])
        inputs = [f"{self.text_column} : {x} Label : " for x in examples[self.text_column]]
        targets = [str(x) for x in examples[self.label_column]]

        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)

        model_inputs, labels = self.pad_tokens(model_inputs, labels, batch_size)
        model_inputs, labels = self.tokens_to_tensors(model_inputs, labels, batch_size)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
    def process_tokens(self):
          
          tokenized_text =  self.dataset.map(self.preprocess,
                                             batched=True,
                                             num_proc=1,
                                             remove_columns=self.dataset["train"].column_names,
                                             load_from_cache_file=False,
                                             desc="Running tokenizer on dataset"
                                            )
          
          return tokenized_text

# COMMAND ----------

dataset = load_dataset("ought/raft", dataset_name)

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]

dataset = dataset.map(lambda x: {"text_label": [classes[label] for label in x["Label"]]},
                      batched=True,
                      num_proc=1)

# COMMAND ----------


preprocessor = PromptTuneTokenize(dataset = dataset, 
                                  text_column = "Tweet text", 
                                  label_column = "text_label", 
                                  max_length = 64, 
                                  tokenizer = "bigscience/bloomz-560m")

processed_datasets = preprocessor.process_tokens()

# COMMAND ----------

model_inputs

# COMMAND ----------

print(
f"Input ids: {model_inputs['train']['input_ids'][0]}\n\n",
f"Attention mask: {model_inputs['train']['attention_mask'][0]}\n\n"
f"Labels: {model_inputs['train']['labels'][0]}"
)

# COMMAND ----------

#train_dataset = model_inputs["train"]
#eval_dataset = model_inputs["train"]

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["train"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

model = get_peft_model(model, peft_config)

print(model.print_trainable_parameters())

# COMMAND ----------

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# COMMAND ----------

model = model.to(device)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

def preprocess_function(examples):
  # Number of input text examples in batch
  batch_size = len(examples[text_column])
  # Formatted string for each observation in the batch
  inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
  # A list of labels in string format
  targets = [str(x) for x in examples[label_column]]

  # Tokenize the model inputs and labels
  model_inputs = tokenizer(inputs)
  labels = tokenizer(targets)

  # Loop through each observation in the batch
  for i in range(batch_size):
        # Grab a single, tokenized input string
        sample_input_ids = model_inputs["input_ids"][i]
        # Grab a single, tokenize label and add the padding token id to the end (end of sequence token id)
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        # Combine the input text and label tokens
        # Examples: [227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 77658, 915, 210, 1936, 106863, 3]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        # Overwrite the labels such that the token id [-100] occures for every token id
        # in the model input that represents the text input, then, add the token ids associated
        # with the labels. This could be used to separate the parts of the model input into 
        # the text and label components
        # Examples: [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1936, 106863, 3]
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        # Create an attention mask sequence of with one entry per model input token
        # Ex. [1,1,1,1,1,1,1,1,1]
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
  return model_inputs 

# COMMAND ----------

processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

# COMMAND ----------

print(
f"Input ids: {processed_datasets['train']['input_ids'][0]}\n",
f"Attention mask: {processed_datasets['train']['attention_mask'][0]}"
)

# COMMAND ----------

def preprocess_function(examples):
  # Number of input text examples in batch
  batch_size = len(examples[text_column])
  # Formatted string for each observation in the batch
  inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
  # A list of labels in string format
  targets = [str(x) for x in examples[label_column]]

  # Tokenize the model inputs and labels
  model_inputs = tokenizer(inputs)
  labels = tokenizer(targets)

  # Loop through each observation in the batch
  for i in range(batch_size):
        # Grab a single, tokenized input string
        sample_input_ids = model_inputs["input_ids"][i]
        # Grab a single, tokenize label and add the padding token id to the end (end of sequence token id)
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        # Combine the input text and label tokens
        # Examples: [227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 77658, 915, 210, 1936, 106863, 3]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        # Overwrite the labels such that the token id [-100] occures for every token id
        # in the model input that represents the text input, then, add the token ids associated
        # with the labels. This could be used to separate the parts of the model input into 
        # the text and label components
        # Examples: [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1936, 106863, 3]
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        # Create an attention mask sequence of with one entry per model input token
        # Ex. [1,1,1,1,1,1,1,1,1]
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

  # Loop through the observations again
  for i in range(batch_size):
      # The combined text and label ids
      sample_input_ids = model_inputs["input_ids"][i]
      # The label ids prefixed with [-100] * len(num_input_text_token_ids)
      label_input_ids = labels["input_ids"][i]

      # Overwrite the model input ids by padding them (as prefixes) to match the length set by max_length; after padding, each sequence of token ids
      # will have a length of 64
      # Example [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 227985, 5484, 915, 2566, 169403, 15296, 36272, 525, ...., 3]
      model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) + sample_input_ids
      # Perform the same transformation on the attention mask, except pad (as prefixes) the attention mask array with 0
      model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
      # Update the labels, -100 prefix padding, to reflect to new padded tokens added to the input_ids
      labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
      # Convert the input ids to tensors, truncating any tokens beyond the max_length parameter
      model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
      # Convert the attention_mask ids to tensors, truncating any tokens beyond the max_length parameter
      model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
      # Convert the label ids to tensors, truncating any tokens beyond the max_length parameter
      labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
  # Add the label ids to the model ouput as a new key
  model_inputs["labels"] = labels["input_ids"]
  
  return model_inputs


# COMMAND ----------

len([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 77658, 915, 210, 1936, 106863, 3])

# COMMAND ----------

processed_datasets = dataset.map(
  preprocess_function,
  batched=True,
  num_proc=1,
  remove_columns=dataset["train"].column_names,
  load_from_cache_file=False,
  desc="Running tokenizer on dataset"
)

# COMMAND ----------

processed_datasets

# COMMAND ----------

print(
f"Input ids: {processed_datasets['train']['input_ids'][0]}\n\n",
f"Attention mask: {processed_datasets['train']['attention_mask'][0]}\n\n"
f"Labels: {processed_datasets['train']['labels'][0]}"
)

# COMMAND ----------


class PromptTuneTokenize:
  def __init__(self, dataset, text_column, label_column, max_length):
    

def preprocess_function(examples):
  # Number of input text examples in batch
  batch_size = len(examples[text_column])
  # Formatted string for each observation in the batch
  inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
  # A list of labels in string format
  targets = [str(x) for x in examples[label_column]]

  # Tokenize the model inputs and labels
  model_inputs = tokenizer(inputs)
  labels = tokenizer(targets)

  # Loop through each observation in the batch
  for i in range(batch_size):
        # Grab a single, tokenized input string
        sample_input_ids = model_inputs["input_ids"][i]
        # Grab a single, tokenize label and add the padding token id to the end (end of sequence token id)
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        # Combine the input text and label tokens
        # Examples: [227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 77658, 915, 210, 1936, 106863, 3]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        # Overwrite the labels such that the token id [-100] occures for every token id
        # in the model input that represents the text input, then, add the token ids associated
        # with the labels. This could be used to separate the parts of the model input into 
        # the text and label components
        # Examples: [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1936, 106863, 3]
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        # Create an attention mask sequence of with one entry per model input token
        # Ex. [1,1,1,1,1,1,1,1,1]
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

  # Loop through the observations again
  for i in range(batch_size):
      # The combined text and label ids
      sample_input_ids = model_inputs["input_ids"][i]
      # The label ids prefixed with [-100] * len(num_input_text_token_ids)
      label_input_ids = labels["input_ids"][i]

      # Overwrite the model input ids by padding them (as prefixes) to match the length set by max_length; after padding, each sequence of token ids
      # will have a length of 64
      # Example [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 227985, 5484, 915, 2566, 169403, 15296, 36272, 525, ...., 3]
      model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) + sample_input_ids
      # Perform the same transformation on the attention mask, except pad (as prefixes) the attention mask array with 0
      model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
      # Update the labels, -100 prefix padding, to reflect to new padded tokens added to the input_ids
      labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
      # Convert the input ids to tensors, truncating any tokens beyond the max_length parameter
      model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
      # Convert the attention_mask ids to tensors, truncating any tokens beyond the max_length parameter
      model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
      # Convert the label ids to tensors, truncating any tokens beyond the max_length parameter
      labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
  # Add the label ids to the model ouput as a new key
  model_inputs["labels"] = labels["input_ids"]
  
  return model_inputs

# COMMAND ----------

# MAGIC %pip install --pre torch torchvision torchaudio --index-url https://pypi.org/simple

# COMMAND ----------

# MAGIC %pip list
