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
- ### preprocess.py: 
    If you want to train the model or predict based on your code comments, you can use `python preprocess.py` to preprocess the dataset. 
- ### codeT5.py: 
    The python file to fine-tune codeT5 for a classifier.
- ### result: 
    This folder contains the training logs, which record the precision, recall  and fscore of the corresponding models. Since github has a limit on the size of files that can be uploaded, the zip files of the 19 trained classifiers and dataset has been uploaded to [Figshare](https://figshare.com/articles/dataset/Classifier_zip/22083500). 
    In [Figshare](https://figshare.com/articles/dataset/java_deprecation/22084031). 
    In order to facilitate the reproduction of our experiments, we also uploaded a classifier separately. [deprecation](https://figshare.com/articles/dataset/java_deprecation/22084031)
    - `.bin` file is the trained model
    - `test_1.gold` is the gold of testing data
    - `test_1.output` is the prediction of the trained model
    - `test_data_of_deprecation.jsonl` is the testing data,  `val_data_of_deprecation.jsonl` is the validation data,`train_data_of_deprecation.jsonl` is the training data
    
    If you can't download the file successfully, you can try to download from other link below.
    - [zenodo](https://zenodo.org/record/7659286#.Y_Q973ZBzEY)：Each classifier can be downloaded separately.
    - [zenodo](https://zenodo.org/record/7659231#.Y_Q_t3ZBzEY): All classifiers are compressed to one file
    
    unzip the zip file:

    - `checkpoint-best-ppl`: the best model on the validation dataset
    - `checkpoint-last`: the last model in the training. We used it to test the testing dataset to calculate the precision, recall, fscore recorded in the paper.
    - the `dev.gold` is the ground truth of the validation dataset; the `dev.output` is the ground truth of the validation dataset in the trainning stage. 
    - the `test_0.gold` is the ground truth of the validation dataset; the `test_0.output` is the ground truth of the validation dataset in the testing stage.
    - the `test_1.gold` is the ground truth of the testing dataset; the `test_1.output` is the prediction of our classifier.
    

- ### model:
  The folder contains the config files for the pre-trained model from the [CodeT5-base](https://huggingface.co/Salesforce/codet5-base/tree/main)

## Experiment
- ### Step 1 : Install the dependency
  transformers 4.17.0  
  torch 1.13.0 \
  python 3.7.11 \
  \
  Download the `added_tokens.json`, `config.json`, `merges.txt`, `pytorch_model.bin`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.json` to the `model` folder from [huggingface](https://huggingface.co/Salesforce/codet5-base/tree/main).
- ### Step 2: Prepare the dataset 
- The data format for training or testing
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
- The data format for prediction 
  ```
  {
  "final_sentence": "i can be used, for example, to delay an action in a data set until currently running animations are complete."
  }
  ```
- you can use `python preprocess.py` to preprocess your own dataset or download our preprocessed dataset, the jsonl files in the folder `dataset`
- ### Step 3: Training the model
    - Train the model on a single GPU/CPU:
    ```commandline
    python codeT5.py --local_rank=-1 --do_train --do_eval --do_test --train_log_filename="java_deprecation" --train_filename="dataset/java/train_data_of_deprecation.jsonl" --dev_filename="dataset/java/val_data_of_deprecation.jsonl" --test_filename="dataset/java/test_data_of_deprecation.jsonl"
    ```
  
    - Distributed training on multi-GPUs: 
    ```commandline
    python -m torch.distributed.launch --nproc_per_node=2 codeT5.py  --do_train --do_eval --do_test --output_dir="java_deprecation_output" --train_log_filename="java_deprecation" --train_filename="dataset/java/train_data_of_deprecation.jsonl" --dev_filename="dataset/java/val_data_of_deprecation.jsonl" --test_filename="dataset/java/test_data_of_deprecation.jsonl"
    ```    
    - `--train_log_filename` refers to the log file when running the experiment. `--output_dir` is the folder used to store the trained model and the prediction. `--train_filename` is the path of training data. `--dev_filename` is the path of validation data.  `--test_filename` is the path of testing data. If `--local_rank`==-1, it will  use a single CPU/GPU in the experiment. `--nproc_per_node` is the number of processes on each machine during distributed training.
  
    - how to use our trained model to test or predict the category of code comments. (choose one command below)
    ```commandline
    python codeT5.py --do_test --test_filename="dataset/java/test_data_of_deprecation.jsonl" --local_rank=-1 --load_model_path="java_deprecation_output/checkpoint-best-ppl/pytorch_model.bin"
  ```
  
  ```commandline
    python -m torch.distributed.launch --nproc_per_node=2 codeT5.py --do_test --output_dir="java_deprecation_output" --train_log_filename="java_deprecation" --load_model_path=commit_files/java_deprecation_output/checkpoint-best-ppl/pytorch_model.bin --test_filename="final_final_dataset/java/test_data_of_deprecation.jsonl"
  ```
- ### Step 4: Result
  The trained model will be stored in the path `--output_dir`. In addition, the `dev_0.output` file is the prediction of validation dataset, the `dev_1.gold` is the ground truth of the validation dataset. `test_<id>` is the result of testing data. The train_`--train_log_filename`.log is the log file which records the number of TP, FP, TN, FN, and `Precision`, `Recall`, `Fscore`.

- ### Final: If you just want to predict
    - Predict the category on a single GPU/CPU:
    ``` commandline
    python codeT5.py --local_rank=-1 --do_pred --load_model_path=java_deprecation_output/checkpoint-best-ppl/pytorch_model.bin --test_filename="dataset/java/test_data_of_deprecation.jsonl"
    ```  

  - Predict the category on multi-GPUs:
  ```commandline
  python -m torch.distributed.launch --nproc_per_node=2 codeT5.py --do_pred --load_model_path=result_non_preprocess/java_deprecation_output/checkpoint-best-ppl/pytorch_model.bin --test_filename="dataset/java/test_data_of_deprecation.jsonl"
   ```
  

  
        


