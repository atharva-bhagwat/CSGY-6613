# (CSGY-6613)AI project: Separating Perception and Reasoning via Relation Networks

- Atharva Bhagwat (acb9244)
- Harini Appansrinivasan (ha1642)
- Abdulqadir Zakir (az2424)

**Goal:**

Visual Question Answering (VQA) is a multi-modal task relating text and images through captions or a questionnaire. For example, with a picture of a busy highway, there could be a question: “How many red cars are there?” or “Are there more motorbikes than cars?”. It is a very challenging task since it requires high-level understanding of both the text and the image and the relationships between them.

In this project, we study the Relation Networks implementations of AI approaches that offer the ability to combine neural and symbolic representations to answer VQA task.

Relation Networks by DeepMind is a simple and representationally flexible general solution to relational reasoning in neural networks. We solve our problem using two different types of data, pixel-based and state-descriptions based.


**Main Files:**
- `setup.md`: Contains the instructions to clone our repo and use github with colab to run our code.
- `run.ipynb`: The main set of commands to run all the scripts.
- `generate_dataset.py`: Used to generate the datasets for both pixel-based and state-description based Sort-of-CELVR. The actual dataset files for both pixel based sort-of-CLEVR and state descriptor based sort-of-CLEVR, ie: `.pkl` files cannot be uploaded to github due to size restrictions. 
- `main.py` & `model.py`: Files related to pixel-based Sort-of-CLEVR data for Q2. Contains the model architecture and the code for training and testing.
- `sd_main.py` & `sd_model.py`: Files related to state-description-based Sort-of-CLEVR data for Q3. Contains the model architecture and the code for training and testing.
- `/sort_of_clevr/sort_of_clevr_descriptor.csv`: For visualization purposes a state descriptor table in `.CSV` format has been added to the repository [here](https://github.com/atharva-bhagwat/CSGY-6613/tree/main/sort_of_clevr).
- `output` folder: All the output plots can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/tree/main/output).
- `model` folder: Contains the best model for pixel-based and state-description based Sort-of-CELVR RN.
- `util.py`: To format the questions for Sort-of-CELVR dataset.


## Describe the RN (20 points)

Description of RN can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/milestone1). The milestone1.ipynb has the detailed architecture and tensor shapes at each stage.

**NOTE:** Our explanation of RN architecture and calculation of tensor-shapes are based on the model parameters as mentioned in the [paper](https://arxiv.org/pdf/1706.01427.pdf). 
The reference repository given in the problem description works with a model with smaller kernels and uses a very different interpretation of questions embeddings, binary string of **length 18**. Whereas, the paper has a binary string of **length 11**.

## VQA on Sort-of-CLEVR (80 points)

### (40 points) Replicate the sort-of-CLEVR dataset result of the paper as quoted in the section 5.3. Please focus only on the CNN, augmented with RN, case.

- With the [github repository](https://github.com/kimhc6028/relational-networks) as reference, we developed the model architecture, keeping the model size smaller than the one given in the paper, but changing the question embedding shape to match that in the paper (length 11).
- We tried many different architectures: by changing the number of units and number of layers in both $g_\theta$ and $f_\phi$. We also tried with different hyper-parameters: learning rates like, $0.0001, 0.01, 0.1$, and optimizers like SGD, Adam.
- Best model architecture: 
  - 4 conv layer block with 24 kernels each with ReLU activation and batch normalization. 
  - $g_\theta$ consists of 4 fully connected layers with 256 units and ReLU activations. 
  - $f_\phi$ consists of 4 fully connected layers with 256 units and ReLU activation and an output layer with 10 units with a softmax activation.
- Best hyper-parameters:
  - number of epochs: 50
  - learning rate: $0.001$
  - Optimizer: Adam
  - Batch size: 64

#### Results:

After training the model for 50 epochs, the best final metrics are as follows:

**Relational Data**

| Train Accuracy(%) | Train Loss | Test Accuracy(%) | Test Loss |
|---|---|---|---|
| 98.40 | 0.045 | 91.28 | 0.304 |

**Non-Relational Data**

| Train Accuracy(%) | Train Loss | Test Accuracy(%) | Test Loss |
|---|---|---|---|
| 99.98 | 0.001 | 99.97 | 0.001 |


Entire training log can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/logs.txt).

Accuracy and Loss plots can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/tree/main/output):

![Accuracy Plot](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/acc.jpg)

![Loss Plot](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/loss.jpg)

Sample test output can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/tree/main/output):

![Test 1](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/test_0.jpg)

![Test 2](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/test_15.jpg)

![Test 3](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/test_30.jpg)

![Test 4](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/test_45.jpg)


### (40 points) Perform the “state description” task and create a table that represents the state of each image.

- State descriptor table (as a CSV file) for the entire dataset can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/sort_of_clevr/sort_of_clevr_descriptor.csv).
- For purpose of training, we use the `.pkl` file, generated using `generate_dataset.py`.

Sample state descriptor for one image:

| color | center | shape|
|---|---|---|
| red | [56, 19] | circle |
| green | [28, 7] | rectangle |
| blue | [6, 34] | rectangle |
| orange | [68, 64] | rectangle |
| gray | [53, 63] | rectangle |
| yellow | [68, 7] | circle |

#### Model architecuture:

The RN network remains the same, only without the convolutional block since we are not dealing with pixels here. It only consists of the fully connected network.

![RN SD](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/docs/RN_sd.jpg)


- We tried many different architectures by changing the number of units and number of layers in both $g_\theta$ and $f_\phi$. We also tried with different sets of hyper-parameters: learning rates like, $0.0001, 0.01, 0.1$, and optimizers like SGD, Adam. But the accuracies in most cases stagnated around $60\\%$.   
- Best model architecture: 
  - $g_\theta$ consists of 3 fully connected layers with 256, 512, and 512 units and ReLU activations. 
  - $f_\phi$ consists of 4 fully connected layers with 512, 512, 512, and 256 units and ReLU activation and an output layer with 10 units with a softmax activation.
- Best hyper-parameters:
  - number of epochs: 50
  - learning rate: $0.001$
  - Optimizer: Adam
  - Batch size: 64

#### Results:

After training the model for 25 epochs, the best final metrics are as follows:

**Relational Data**

| Train Accuracy(%) | Train Loss | Test Accuracy(%) | Test Loss |
|---|---|---|---|
| 69.02 | 0.575 | 68.25 | 0.595 |

**Non-Relational Data**

| Train Accuracy(%) | Train Loss | Test Accuracy(%) | Test Loss |
|---|---|---|---|
| 75.38 | 0.348 | 75.45 | 0.345 |

Entire training log can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/logs_sd.txt).

Accuracy and Loss plots can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/tree/main/output):

![Accuracy Plot](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/acc_sd.jpg)

![Loss Plot](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/loss_sd.jpg)
