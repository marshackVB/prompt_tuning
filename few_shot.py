# Databricks notebook source
# MAGIC %md ### ToDo:  
# MAGIC
# MAGIC 1. Download the banking77 dataset
# MAGIC 2. Select a subject of categories, for instance, 20 of them, to make prompt engineering (describing the categories) easier. 
# MAGIC 3. Download a model and create a few shot prompt to classify the evaluation dataset (really, treating this as a test dataset).
# MAGIC 4. Classify the dataset and capture F1, precision, and recall scores.
# MAGIC 5. Prompt-tune the model using the training dataset (splits out an evaluation dataset)
# MAGIC 6. Classify the test dataset capturing the same validation statistics to compare the models.
# MAGIC 7. Try another variation of the analysis that assumes only a small labeld training dataset exists, for instance, 5 observations per category.
# MAGIC 8. Traing another prompt tuning model, spliting this small dataset into training and validation datasets.
# MAGIC 9. Predict on the full test dataset.
# MAGIC 10. Use the small training dataset and predictions from the large test dataset to create a golden dataset.
# MAGIC 11. Traing a very small transformer model, like distilbert-base-uncase, on the golden dataset, spliting into training, validation and test datasets.
# MAGIC 12. Calculate perfomance on the test dataset
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Notes:
# MAGIC  - Could be interesting to compare prompt design (making the category descriptions better) vs. prompt tuning (keeping the prompt in a basic form and using it to fine tune). Which approach yields better results? Prompt tuning is certaining easier and less subjective.
# MAGIC
# MAGIC  - Try both training at full precision, 16 bit, and 8 bit.
# MAGIC  - Try inference at both full precision, 16 bit, and 8 bit.
# MAGIC  - Compare predictive performance, training and inference times, and memory footprint across all precisions.
# MAGIC  - For very small training datasets, such as 5 observations per category, are large, prompt-tuned LLMs for effective?

# COMMAND ----------

# MAGIC %pip install evaluate einops

# COMMAND ----------

from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          default_data_collator, 
                          get_linear_schedule_with_warmup,
                          TrainingArguments,
                          Trainer,
                          pipeline,
                          EarlyStoppingCallback,
                          BitsAndBytesConfig)

from peft import (get_peft_config, 
                  get_peft_model, 
                  PromptTuningInit, 
                  PromptTuningConfig, 
                  TaskType, 
                  PeftType, 
                  PeftConfig, 
                  PeftModel)

import evaluate
                  
import torch
from datasets import load_dataset, Dataset, DatasetDict
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessor import PromptTuneTokenize

from prompts import banking_77_prompt

# COMMAND ----------

banking_77 = load_dataset("banking77")

# COMMAND ----------

to_rename = {"label": "label_id"}

train = pd.DataFrame(banking_77['train'])
train.rename(columns=to_rename, inplace=True)

# COMMAND ----------

train.head()

# COMMAND ----------

device = 'cuda'

to_rename = {"label": "label_id"}

train = pd.DataFrame(banking_77['train'])
train.rename(columns=to_rename, inplace=True)
test = pd.DataFrame(banking_77['test'])
test.rename(columns=to_rename, inplace=True)

include_labels = [0,1,2,3,4,5,6,7,8,9,10,11]

label_map = {0: "Activate my card",
             1: "Age limit question",
             2: "Google Pay or Apple Pay support",
             3: "ATMs support",
             4: "Automatic top up",
             5: "Balance not updated after bank transfer",
             6: "Balance not updated after cheque or cash deposit",
             7: "Beneficiary is not allowed",
             8: "Cancel a transaction",
             9: "Card is about to expire",
             10: "Where are cards accepted",
             11: "Delivery of new card or when will card arrive"}

label_to_id = {label: id for id, label in label_map.items()}

train = train[train.label_id.isin(include_labels)] 
test = test[test.label_id.isin(include_labels)] 

train["label"] = train.label_id.apply(lambda x: label_map[x])
test["label"] = test.label_id.apply(lambda x: label_map[x])

train = Dataset.from_pandas(train)
test = Dataset.from_pandas(test)

dataset = DatasetDict({"train": train, "test": test})

# COMMAND ----------

#model_name_or_path = "bigscience/bloomz-560m"
model_name_or_path = "bigscience/bloomz-1b7"
#model_name_or_path = "bigscience/bloomz-3b"
#model_name_or_path = "mosaicml/mpt-7b-instruct"

preprocessor = PromptTuneTokenize(dataset = dataset, 
                                  text_column = "text", 
                                  label_column = "label", 
                                  max_length = 64, # This could likely be shorter
                                  tokenizer = model_name_or_path)

cols_to_remove = ["__index_level_0__", 'text', 'label', 'label_id']

processed_datasets = preprocessor.process_tokens(cols_to_remove)
processed_datasets

# COMMAND ----------

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["test"]

train_dataset = train_dataset.shuffle(seed=42)

# COMMAND ----------

prompt_tuning_init_text = """

Classify banking customer questions into one of the category below. Choose
the most simiar category and do not make any additional categories.


Categories: 

Activate my card

Age limit question

Google Pay or Apple Pay support

ATMs support

Automatic top up

Balance not updated after bank transfer

Balance not updated after cheque or cash deposit

Beneficiary is not allowed

Cancel a transaction

Card is about to expire

Where are cards accepted

Delivery of new card or when will card arrive

"""

peft_config = PromptTuningConfig(task_type = TaskType.CAUSAL_LM,
                                 prompt_tuning_init=PromptTuningInit.TEXT,
                                 num_virtual_tokens=8,
                                 prompt_tuning_init_text=prompt_tuning_init_text,
                                 tokenizer_name_or_path = model_name_or_path
                                )

# COMMAND ----------

def init_model(peft_config=peft_config):
  device = "cuda"
  model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", load_in_8bit=True)
  model = get_peft_model(model, peft_config)
  return model.to(device)

model = init_model()

training_args = {"output_dir": "/checkpoints",
                 "overwrite_output_dir": True,
                 "per_device_train_batch_size": 16,
                 "per_device_eval_batch_size": 16,
                 "weight_decay": 0.01,
                 "num_train_epochs": 7,
                 "save_strategy": "epoch", 
                 "evaluation_strategy": "epoch",
                 "logging_strategy": "epoch",
                 "load_best_model_at_end": False, # Throughs an error during 8-bit training
                 "save_total_limit": 2,
                 #"metric_for_best_model": "eval_loss", #https://discuss.huggingface.co/t/early-stopping-training-using-validation-loss-as-the-metric-for-best-model/31378
                 #"greater_is_better": False,
                 "seed": 123,
                 "report_to": "none",
                 "gradient_accumulation_steps": 1,
                 #"fp16": True,
                 "learning_rate": 3e-2}

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.05)

trainer = Trainer(model = model,
                  args = TrainingArguments(**training_args),
                  train_dataset = train_dataset, 
                  eval_dataset = eval_dataset)
                  #callbacks=[early_stopping_callback])

trainer.train()

# COMMAND ----------

# MAGIC %sh ls /checkpoints/checkpoint-707

# COMMAND ----------

# MAGIC %sh mkdir /peft_config

# COMMAND ----------

model.save_pretrained('/peft_config')

# COMMAND ----------

# MAGIC %sh ls /peft_config

# COMMAND ----------

# MAGIC %md Load model

# COMMAND ----------

config = PeftConfig.from_pretrained('/peft_config')
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", load_in_8bit=True)
model = PeftModel.from_pretrained(model, '/peft_config')

model.to(device)

# COMMAND ----------

# MAGIC %md Evaluation

# COMMAND ----------


def config_prediction_func(label_to_id, text_column, unknown_label_id=-99):
  """
  If the model is generated unknown labels, consider adjusting the prompt to
  be more descriptive.
  """

  def get_predictions(batch):

    formatted_text = [f'{text_column} : {text} Label : ' for text in batch[text_column]]

    tokens = preprocessor.tokenizer(formatted_text, padding='longest', truncation=True, max_length=64, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        prediction = model.generate(input_ids = input_ids,
                                    attention_mask = attention_mask,
                                    max_new_tokens=10, 
                                    eos_token_id=3
                      )
        
        predicted_labels = preprocessor.tokenizer.batch_decode(prediction.detach().cpu().numpy(), skip_special_tokens=True)
        predicted_labels = [predicted_label.split(":")[-1].strip() for predicted_label in predicted_labels]
        predicted_label_id = [label_to_id.get(predicted_label, unknown_label_id) for predicted_label in predicted_labels]

    return {"predicted_label": predicted_labels, "predicted_label_id": predicted_label_id}
  
  return get_predictions

# COMMAND ----------

get_predictions = config_prediction_func(label_to_id, 'text')

eval_predictions = test.map(get_predictions, batched=True, batch_size=12)

# COMMAND ----------

print(classification_report(eval_predictions["label_id"], eval_predictions["predicted_label_id"]))

# COMMAND ----------

print(classification_report(eval_predictions["label_id"], eval_predictions["predicted_label_id"]))

# COMMAND ----------

for row in eval_precictions:
  if row['predicted_label_id'] == -99:
    print(row['text'], row['predicted_label'])

# COMMAND ----------

print(classification_report(eval_predictions["label_id"], eval_precictions["predicted_label_id"]))

# COMMAND ----------

# MAGIC %md Old code

# COMMAND ----------

print(classification_report(eval_precictions["label_id"], eval_precictions["predicted_label_id"]))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

def get_predictions(batch):
  label_to_id = {label: id for id, label in label_map.items()}

  input_ids = torch.tensor(batch['input_ids']).to(device)
  attention_mask = torch.tensor(batch['attention_mask']).to(device)

  with torch.no_grad():
      prediction = model.generate(input_ids = input_ids,
                                  attention_mask = attention_mask,
                                  max_new_tokens=10, 
                                  eos_token_id=3
                    )
      
      predicted_labels = preprocessor.tokenizer.batch_decode(prediction.detach().cpu().numpy(), skip_special_tokens=True)
      predicted_labels = [predicted_label.split(":")[-1].strip() for predicted_label in predicted_labels]
      predicted_label_id = [label_to_id.get(predicted_label) for predicted_label in predicted_labels]

  return {"predicted_label": predicted_labels, "predicted_label_id": predicted_label_id}

# COMMAND ----------

pred = x.map(get_predictions, batched=True, batch_size=16)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

sample = x.select(range(10))
label_to_id = {label: id for id, label in label_map.items()}

input_ids = torch.tensor(sample['input_ids']).to(device)
attention_mask = torch.tensor(sample['attention_mask']).to(device)

with torch.no_grad():
    prediction = model.generate(input_ids = input_ids,
                                 attention_mask = attention_mask,
                                 max_new_tokens=10, 
                                 eos_token_id=3
                  )
    
    predicted_labels = preprocessor.tokenizer.batch_decode(prediction.detach().cpu().numpy(), skip_special_tokens=True)
    predicted_labels = [predicted_label.split(":")[-1].strip() for predicted_label in predicted_labels]
    predicted_label_id = [label_to_id.get(predicted_label) for predicted_label in predicted_labels]


print(predicted_labels, predicted_label_id)


# COMMAND ----------

label_to_id = {label: id for id, label in label_map.items()}

# COMMAND ----------

def get_predictions(batch):

  input_ids = torch.tensor(batch['input_ids']).to(device)
  attention_mask = torch.tensor(batch['attention_mask']).to(device)

  with torch.no_grad():
    return model.generate(input_ids = input_ids,
                                 attention_mask = attention_mask,
                                 max_new_tokens=10, 
                                 eos_token_id=3
                  )


pred = x.map(get_predictions, batched=True, batch_size=12)

# COMMAND ----------



# COMMAND ----------

preprocessor.tokenizer(x['formatted_text'][:10], padding=True, truncation=True, return_tensors="pt")

# COMMAND ----------

x['formatted_text']

# COMMAND ----------

test

# COMMAND ----------

def get_predictions(batch):

  text_column = 'text'

  inputs = preprocessor.tokenizer(
    f'{text_column} : {"@nationalgridus I have no water and the bill is current and paid. Can you do something about this?"} Label : ',
    return_tensors="pt",
)

# COMMAND ----------



# COMMAND ----------

eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# COMMAND ----------

samples = eval_dataset.select(range(10))

# COMMAND ----------

samples[0]['attention_mask'].dim()

# COMMAND ----------

samples[0]['input_ids'].dim()

# COMMAND ----------

samples[0]

# COMMAND ----------

obs_1 = eval_dataset[0]
input_ids = torch.tensor(obs_1['input_ids']).to(device)
attention_mask = torch.tensor(obs_1['attention_mask']).to(device)

with torch.no_grad():
  model.generate(input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens=10, 
                eos_token_id=3
                )

# COMMAND ----------



# COMMAND ----------

samples = eval_dataset.select(range(1))

input_ids = torch.tensor(samples[0]['input_ids']).to(device)
attention_mask = torch.tensor(samples[0]['attention_mask']).to(device)

model.generate(input_ids = input_ids,
                           attention_mask = attention_mask,
                           max_new_tokens=10, 
                           eos_token_id=3
                          )

# COMMAND ----------

samples = eval_dataset.select(range(1))

for sample in samples:

  input_ids = sample['input_ids'].to(device)
  attention_mask = sample['attention_mask'].to(device)

  print(input_ids)
  print(attention_mask)

  #torch_inputs = {k: v.to(device) for k, v in sample.items()}

  with torch.no_grad():

    outputs = model.generate(input_ids = input_ids,
                             attention_mask = attention_mask,
                             max_new_tokens=10, 
                            eos_token_id=3
                          )

# COMMAND ----------



# COMMAND ----------

preprocessor.tokenizer.eos_token_id

# COMMAND ----------

samples = eval_dataset.select(range(10))

torch_inputs = {k: torch.tensor(v).to(device) for k, v in samples.items()}

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

inputs = {k: torch.tensor(v).to(device) for k, v in samples[0].items()}

# COMMAND ----------

inputs

# COMMAND ----------

device = 'cuda'

def get_predictions(inputs):

  predictions = []
  torch_inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}

  for input in torch_inputs:

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    #input_ids = torch.tensor([input['input_ids']]).to(device)
    #attention_mask = torch.tensor([input['attention_mask']]).to(device)

    with torch.no_grad():

      outputs = model.generate(
          input_ids=input_ids, 
          attention_mask=attention_mask, 
          max_new_tokens=10, 
          eos_token_id=preprocessor.tokenizer.pad_token_id
      )

      predicted_labels = preprocessor.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    #predicted_labels = predicted_labels[0].split(":")[-1].strip()

    predictions.append(predicted_labels)

  return {"predicted_label": predictions}

predictions = eval_dataset.map(get_predictions, batched=True, batch_size=16)

# COMMAND ----------

device = 'cuda'

def get_predictions(inputs):

  input_ids = torch.tensor(inputs['input_ids']).to(device)
  attention_mask = torch.tensor(inputs['attention_mask']).to(device)

  with torch.no_grad():

    outputs = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=10, 
        eos_token_id=3
    )

    predicted_labels = preprocessor.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    predicted_labels = [predicted_label.split(":")[-1].strip() for predicted_label in predicted_labels]

  return {"predicted_label": predicted_labels}

predictions = eval_dataset.map(get_predictions, batched=True, batch_size=16)

# COMMAND ----------

predictions['predicted_label']

# COMMAND ----------

eval_dataset

# COMMAND ----------



# COMMAND ----------

ds = dataset['test'].select(range(10))

for row in ds:
  print(row['text'])

# COMMAND ----------

# MAGIC %md Predict on evaluation dataset

# COMMAND ----------



# COMMAND ----------

#eval_set = dataset['test'].select(range(10))
eval_set = dataset['test']
text_col = "text"
label_col = "label"
label_id_col = "label_id"
device='cuda'
predictions = []
label_to_id = {label: id for id, label in label_map.items()}

for row in eval_set:

  text = row[text_col].strip()
  label = row[label_col].strip()
  label_id = row[label_id_col]

  inputs = preprocessor.tokenizer(
      f'text : {text} Label : ',
      return_tensors="pt",
  )

  with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
    )

  predicted_label = preprocessor.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
  predicted_label = predicted_label[0].split(":")[-1].strip()
  predicted_label_id = label_to_id.get(predicted_label, -99)

  correct_prediction = 1 if predicted_label == label else 0

  predictions.append({"text": text, 
                      "label": label,
                      "label_id": label_id,
                      "prediction": predicted_label,
                      "prediction_label_id": predicted_label_id,
                      "correct_prediction": correct_prediction})

# COMMAND ----------

predictions

# COMMAND ----------

labels = [prediction['label_id'] for prediction in predictions]
prediction_label_id = [prediction['prediction_label_id'] for prediction in predictions]

# COMMAND ----------

print(classification_report(labels, prediction_label_id))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

eval_set = dataset['test'].select(range(10))
text_col = "text"
label_col = "label"
label_id_col = "label_id"
device='cuda'
predictions = []

for row in eval_set:

  text = row[text_col].strip()
  label = row[label_col].strip()
  label_id = row[label_id_col]

  inputs = preprocessor.tokenizer(
      f'text : {text} Label : ',
      return_tensors="pt",
  )

  with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
    )

  predicted_label = preprocessor.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
  predicted_label = predicted_label[0].split(":")[-1].strip()

  correct_prediction = 1 if predicted_label == label else 0

  predictions.append({"text": text, 
                      "label": label,
                      "label_id": label_id,
                      "prediction": predicted_label,
                      "correct_prediction": correct_prediction})

# COMMAND ----------

[prediction['correct_prediction'] for prediction in predictions]

# COMMAND ----------

classification_report()

# COMMAND ----------

# MAGIC %md Old code

# COMMAND ----------

predictions = trainer.predict(eval_dataset.select(range(10)))

# COMMAND ----------

evaluate.list_evaluation_modules()

# COMMAND ----------

def compute_metrics(eval_preds):
    metric = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# COMMAND ----------

def evaluate(dataset, tokenizer, model, text_col, label_col):



  inputs = tokenizer(
    f'text : {text} Label : ',
    return_tensors="pt",
)


# COMMAND ----------

batched=True
batch_size=16

# COMMAND ----------

dataset['test']

# COMMAND ----------

eval_dataset

# COMMAND ----------

inputs = tokenizer(
    f'text : {text} Label : ',
    return_tensors="pt",
)

with torch.no_grad():
  inputs = {k: v.to(device) for k, v in inputs.items()}
  outputs = model.generate(
      input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
  )

print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

# COMMAND ----------

# MAGIC %md Load Model

# COMMAND ----------

double_quant_config = BitsAndBytesConfig(
   #load_in_4bit=True
   load_in_8bit=True
)

config = PeftConfig.from_pretrained('/peft_config')
#quantization_config=double_quant_config
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
model = PeftModel.from_pretrained(model, '/peft_config')

device = 'cuda'
model.to(device)

# COMMAND ----------

test['text'][:10]

# COMMAND ----------

text = test['text'][104]

inputs = tokenizer(
    f'text : {text} Label : ',
    return_tensors="pt",
)

with torch.no_grad():
  inputs = {k: v.to(device) for k, v in inputs.items()}
  outputs = model.generate(
      input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
  )

print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

# COMMAND ----------

Activate my card
Age limit question
Google Pay or Apple Pay support
ATMs support
Automatic top up
Balance not updated after bank transfer
Balance not updated after cheque or cash deposit
Beneficiary is not allowed
Cancel a transaction
Card is about to expire
Where are cards accepted
Delivery of new card or when will card arrive

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



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

device = "cuda"
#model_name_or_path = "bigscience/bloomz-560m"
model_name_or_path = "bigscience/bloomz-1b7"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

def init_model():
  model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", load_in_8bit=True)
  return model

model = init_model()

# COMMAND ----------

banking_77_prompt = """
You must classify questions from bank customer into one of the Categories below.


Categories: 

Activate my card

Age limit question

Google Pay or Apple Pay support

ATMs support

Automatic top up

Balance not updated after bank transfer

Balance not updated after cheque or cash deposit

Beneficiary is not allowed

Cancel a transaction

Card is about to expire

Where are cards accepted

Delivery of new card or when will card arrive

Linking card to app

Card is not working

Card payment fee charged

Card payment wrong exchange rate

Card swallowed or not returned by ATM machine

Cash withdraw fee charged

Card payment not recognized

Change PIN number

Contactless payment not working

Countries where card is supported

Cash withdraw was declined

Transfer was declined

Edit personal details


User: Can you help me activate my card?
AI: Activate my card

User: My card gets rejected when I try to use it
AI: Card is not working

User: {question}
AI:

"""

question = "How can I order a card?"
prompt = banking_77_prompt.format(question=question)

# COMMAND ----------

inputs = tokenizer(
prompt,
    return_tensors="pt",
)

with torch.no_grad():
  inputs = {k: v.to(device) for k, v in inputs.items()}
  outputs = model.generate(
      input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
  )

result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
print(result[0])

# COMMAND ----------

outputs.shape
