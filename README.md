# (CSGY-6613)AI project: Separating Perception and Reasoning via Relation Networks

- Atharva Bhagwat (acb9244)
- Harini Appansrinivasan (ha1642)
- Abdulqadir Zakir (az2424)

**NOTE:**
- The dataset files for both image based sort-of-CLEVR and state descriptor based sort-of-CLEVR, ie: `.pkl` files cannot be uploaded to github due to size restrictions. These files can be generated using `generate_dataset.py`.
- For visualization purposes a state descriptor table in `.CSV` file is added in the repository.
- Output plots can be found in the [output folder](https://github.com/atharva-bhagwat/CSGY-6613/tree/main/output).
- `run.ipynb` is the main file used to run all the scripts.

## Describe the RN (20 points)

Description of RN can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/milestone1/milestone1.ipynb).

**NOTE:** Above description, shape calculation is according to model parameters as mentioned in the [paper](https://arxiv.org/pdf/1706.01427.pdf). The reference repository given in the problem description works with a model with smaller kernels and a very different interpretation of questions embeddings, binary string of **length 18**. Whereas, the paper has a binary string of **length 11**.

## QA on Sort-of-CLEVR (80 points)

### (40 points) Replicate the sort-of-CLEVR dataset result of the paper as quoted in the section 5.3. Please focus only on the CNN, augmented with RN, case.

- With the [github repository](https://github.com/kimhc6028/relational-networks) as reference, we developed the model architecture, keeping the model size smaller but changing the question embedding shape, making it same as the paper(length 11).
- Best model parameters: 4 conv layer block with 24 kernels each with ReLU activation and batch normalization. $g_\theta$ consists of 4 fully connected layers with 256 units and ReLU activations. $f_\phi$ consists of 4 fully connected layers with 256 units and ReLU activation and an output layer with 10 units with a softmax activation. We use Adam optimizer with learning rate of $0.001$.
- We tried many different architectures by changing the number of units and number of layers in both $g_\theta$ and $f_\phi$, and learning rates like, $0.0001, 0.01, 0.1$.
- After training the model for 50 epochs, the best final metrics are as follows:

**Relational Data:**

On training set:

| Accuracy(%) | Loss |
|---|---|
| 98.40 | 0.045 |

On testing set:

| Accuracy(%) | Loss |
|---|---|
| 91.28 | 0.304 |

**Non-Relational Data:**

On training set:

| Accuracy(%) | Loss |
|---|---|
| 99.98 | 0.001 |

On testing set:

| Accuracy(%) | Loss |
|---|---|
| 99.97 | 0.001 |

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

State descriptor table(as a CSV file) for the entire dataset can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/sort_of_clevr/sort_of_clevr_descriptor.csv).
For purpose of training, we use the `.pkl` file, generated using `generate_dataset.py`.

Sample state descriptor for one image:

| color | center | shape|
|---|---|---|
| red | [56, 19] | circle |
| green | [28, 7] | rectangle |
| blue | [6, 34] | rectangle |
| orange | [68, 64] | rectangle |
| gray | [53, 63] | rectangle |
| yellow | [68, 7] | circle |

#### Model architecuture
The RN network remains the same, only without the convolutional block. ie: it only consists of the fully connected network.

![RN SD](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/docs/RN_sd.jpg)

- Best model parameters: $g_\theta$ consists of 3 fully connected layers with 256, 512, and 512 units and ReLU activations. $f_\phi$ consists of 4 fully connected layers with 512, 512, 512, and 256 units and ReLU activation and an output layer with 10 units with a softmax activation. We use Adam optimizer with learning rate of $0.001$.
- We tried many different architectures by changing the number of units and number of layers in both $g_\theta$ and $f_\phi$, and learning rates like, $0.0001, 0.01, 0.1$. But the accuracies would stagnate around $60\%$.
- After training the model for 25 epochs, the best final metrics are as follows:

**Relational Data:**

On training set:

| Accuracy(%) | Loss |
|---|---|
| 98.40 | 0.045 |

On testing set:

| Accuracy(%) | Loss |
|---|---|
| 91.28 | 0.304 |

**Non-Relational Data:**

On training set:

| Accuracy(%) | Loss |
|---|---|
| 99.98 | 0.001 |

On testing set:

| Accuracy(%) | Loss |
|---|---|
| 99.97 | 0.001 |

Accuracy and Loss plots can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/tree/main/output):

![Accuracy Plot](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/acc_sd.jpg)

![Loss Plot](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/loss_sd.jpg)
