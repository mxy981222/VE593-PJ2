## VE593 Project2

### Project description

In this project we are required to model BN and DBN respectively to diagnose the disease named nuc and predict the trend of stock market, given the specific dataset.

## System requirement

Program and run in Jupyternote book with python 3.8 and

```python
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
```

To meet the system requirement, you should have

```shell
$conda install graphviz
$pip install graphviz
$pip install pydotplus
```

Also, you should add the directory of the #bin of #graphviz into PATH.

## Part1. BN

The file is named as $protain.py$ there are several functions within it.

1. `partition(filename)`
  
  this function is used to generate the test file and train file.
  
  ```
  input: filename
  output: separate the filename.csv into filename_train.csv and filename_test.csv, with size 7:3 
  ```
  
2. `main()`
  
  this function will directly train the model with three different methods and compare the result of two of them.
  
  ```
  input: none
  output: accuracy of the two model
  ```
  

## Part2. DBN

The file is named as $dbn.py$ there are different methods within it

1. `genre(s)`
  
  ```
  input: a list of number s
  output: a list of return of s
  ```
  
2. `genbin(l)` and `genbinv(v)`
  
  ```
  input: a list of return
  output: a discretized variable of the input, and the number of variables after discretization
  ```
  
3. `gentt(filename)`
  
  It will generate the train and test data
  
  ```
  input: the name of the .csv file
  output: the number of variables of each feature, filename_train.csv and filename_test.csv
  ```
  
4. `trainmodel(filename)`
  
  It can train the 1-order markov chain with the given data.
  
  ```
  input: the name of the file
  output: the number of prediction to go up, and the number of the correct predictions
  ```
  
5. `genttk(filename,k)`
  
  It will generate the train and test data
  
  ```
  input: the name of the .csv file, the order k of the model
  output: the number of variables of each feature, filename_traink.csv and filename_testk.csv
  ```
  
6. `kmodel(filename,k)`
  
  It can train the k-order markov chain with the given data.
  
  ```
  input: the name of the file, the order k of the model
  output: the number of prediction to go up, and the number of the correct predictions
  ```
  
7. `main()`
  
  It can will use $kmodel(filename,k)$ to train 1 to 20 order markov chain for prediction and evaluate the results and also plot the evaluation.
  
  ```
  output: a plot of evaluation of different order 
  ```
