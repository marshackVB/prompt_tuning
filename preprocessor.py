from transformers import AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
import torch


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
    Note:
    This should probably be swapped out for dynamic padding: https://huggingface.co/docs/transformers/tasks/language_modeling#preprocess
    Iterate through every observation in the batch of tokenized texts and labels. Extend the lengths of the
    input_ids, attention_mask, and labels arrays to the max_length. For input_ids, prepend the sequence using
    the tokenizer's pad_token_id. For attention_mask, prepend the sequence using 0. For labels, prepend using
    the value [-100].Finally, convert the arrays to tensors, recoming any tokens ids that exceed the max_length.

    Note that for decoder-only architectures, left-side padding should be used.

    Note that tokens set to -100 are ignored when calculating the loss. For examples, see the Bloom models
    documentation: https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/bloom#transformers.BloomForCausalLM.forward.labels
                   https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling.mlm

                   See the Pytorch DataCollatorForLanguageModeling example at https://huggingface.co/docs/transformers/tasks/language_modeling#preprocess
  

    Example for a single observation - all arrays are of length max_length:

    input_ids (3 is the model's pad token id in this case):
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
     3, 3, 3, 3, 3, 3, 3, 3, 227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 
     77658, 915, 210, 1936, 106863, 3]

    attention_mask (0 indicates tokens that are not part of the model input text or labels):
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    labels (-100 prevents input text tokens from being considered during the loss calculation; we care only about
            the loss on the labels):
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


  def process_tokens(self, cols_to_remove=None):
        
    tokenized_text =  self.dataset.map(self.preprocess,
                                        batched=True,
                                        num_proc=1,
                                        remove_columns = cols_to_remove if cols_to_remove else [],
                                        load_from_cache_file=False,
                                        desc="Running tokenizer on dataset"
                                      )
        
    return tokenized_text
  


def config_prediction_func(label_to_id, text_column, unknown_label_id=-99):
  """
  Configure a function that tokenizer input texts and generates predictions. Map
  the generted text prediction back to a label id. If the model is generated unknown 
  labels, consider adjusting the prompt to be more descriptive.
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



"""
train = pd.DataFrame(dataset['train'])
test = pd.DataFrame(dataset['test'])
train_test = pd.concat([train, test])

label_col = "text_label"
other_cols = [col for col in train_test.columns if col != label_col]

X_train, X_test, y_train, y_test = train_test_split(train_test[other_cols], train_test[label_col], test_size=0.2, random_state=42)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train = Dataset.from_pandas(train)
test = Dataset.from_pandas(test)

dataset = DatasetDict(
    {
        "train": train,
        "test": test,
    }
)
"""