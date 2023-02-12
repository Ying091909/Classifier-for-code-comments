## Classifying Code Comments via Pre-trained Programming Language Model


## Content
- ### arff: 
    the `.arff` files with TF_IDF and NLP features extracted from the comment sentences derived from the  Rani et al. in the paper [How to identify class comment types? A multi-language approach for class comment classification](https://www.sciencedirect.com/science/article/pii/S0164121221001448)
- ### dataset:
  - #### `pre_<language>.csv`
    the file contains the data after the first stage of pre-processing
  - #### java: 
    - `train_java_of_<category>.csv` is the file which contains the training data of `<category>` in java.
    - `val_java_of_<category>.csv` is the file which contains the validation  data of `<category>` in java.
    - `test_java_of_<category>.csv` is the file which contains the testing data of `<category>` in java.
    - The jsonl file such as`<type>_java_of_<category>.jsonl` is the data file which was used to fine-tune CodeT5.
  - #### pharo: Same structure as Java.
  - #### python: Same structure as Java.
- ### preprocess.py: the python file to preprocess the dataset.
- ### codeT5.py: the python file to fine-tune codeT5 for a binary classifier.
- ### result: the folder contains the training log and the result. The trained model were uploaded. 
- ### model: the folder contains the pre-trained model and tokenizer from the [CodeT5-base](https://huggingface.co/Salesforce/codet5-base/tree/main)

## Tips
- ### Dependency
  transformers 4.10.0 \
  torchvision 0.14.0\
  torch 1.13.0 
- ### The data format for training 
  ```
  {
  "comment_sentence_id": 378,
  "class": "BlInfiniteItemAnimationsFinished",
  "comment_sentence": "i can be used, for example, to delay an action in a data set until currently running animations are complete.",
  "partition": 1,
  "instance_type": 0,
  "category": "Example",
  }
  ```
- ### The data format for prediction 
  ```
  {
  "final_sentence": "i can be used, for example, to delay an action in a data set until currently running animations are complete."
  }
  ```
- ### Train
    - Train the model on a single GPU/CPU:\
    `python codeT5.py --local_rank=-1 --do_train --do_eval --do_test --train_log_filename="java_deprecation" --train_filename="dataset/java/train_data_of_deprecation.jsonl" --dev_filename="dataset/java/val_data_of_deprecation.jsonl" --test_filename="dataset/java/test_data_of_deprecation.jsonl"
`
  
    - Distributed training on multi-GPUs: \
    `python -m torch.distributed.launch --nproc_per_node=2 codeT5.py  --do_train --do_eval --do_test --output_dir="java_deprecation_output" --train_log_filename="java_deprecation" --train_filename="dataset/java/train_data_of_deprecation.jsonl" --dev_filename="dataset/java/val_data_of_deprecation.jsonl" --test_filename="dataset/java/test_data_of_deprecation.jsonl"
`    
 
- ### Prediect
    - Predict the category on a single GPU/CPU:\
    ` python codeT5.py --local_rank=-1 --do_pred --load_model_path=java_deprecation_output/checkpoint-best-ppl/pytorch_model.bin --test_filename="dataset/java/test_data_of_deprecation.jsonl"`  

  - Predict the category on multi-GPUs:\
  `python -m torch.distributed.launch --nproc_per_node=2 codeT5.py --do_pred --load_model_path=result_non_preprocess/java_deprecation_output/checkpoint-best-ppl/pytorch_model.bin --test_filename="dataset/java/test_data_of_deprecation.jsonl"
`

  
        


