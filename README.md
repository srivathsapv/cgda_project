# Computational Genomics Final Project - Spring 2019

## Team Members
1. Ayush Agarwal (aagarw33)
2. Lohita Sivaprakasam (lsivapr1)
3. Satish Palaniappan (spalani2)
4. Srivathsa Pasumarthi (psrivat1)

## Steps to Run the code

**Important Note**: As required by the project description, we do not have a `run.sh` file as we have many modes of running our
train and test scripts. Hence we have given detailed steps to run our code. If you want to have a look at the hyperparameter
configuration for all our models, you can find it in `config/hyperparams.json`

1. Set the python path to the root directory of the project folder.
```
export PYTHONPATH=$PYTHONPATH:<root_path_of_project_folder>
```

2. Install `graphviz` using the following command

for Mac OSX
```
brew install graphviz
```

for Ubuntu
```
sudo apt-get install graphviz
```

3. Install requirements listed in the requirements.txt file.

```
pip3 install -r requirements.txt
```

4. Run data pre-processing step:

```
python3 process_data.py
```
Once preprocess is run, the data will be generated in the `data` folder.

5. Run the model training script


```
python3 train_model.py --model-name <name_of_the_model> --is-demo
```

where `<name_of_the_model>` can be one of the following
```
'basic_kmer', 'basic_vector', 'basic_onehot', 'cnn_qrcode', 'rnn', 'lstm_vae_ordinal', 'lstm_vae_kmer_4', 'lstm_vae_kmer_5',
'hybrid_vae_ordinal', 'hybrid_vae_kmer_4', 'hybrid_vae_kmer_5'
```

Example:
```
python3 train_model.py --model-name cnn_qrcode --is-demo
```

The `is-demo` flag is given as a convenience to finish the training process quickly just as a method to verify that the code
is running without any bugs. In the demo mode, model training is run only for a very few epoch/iterations (usually 1 or 2). The
training artifacts (like model weights, accuracy curves etc) are stored in the `results` directory.

If you want the entire model training to happen, then run the model without the `is-demo` flag like shown below

```
python3 train_model.py --model-name cnn_qrcode
```

6. Run the model testing script

```
python3 test_model.py --model-name <name_of_the_model>
```

where `<name_of_the_model>` can be one from the above list. Accuracy metrics and progress will be displayed while the results
will be stored according to the model in the `results` folder. Explore and enjoy! :)

Example:
```
python3 test_model.py --model-name cnn_qrcode
```
