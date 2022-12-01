# CSGY-6613
This repo contains code for AI project: Separating Perception and Reasoning via Relation Networks

## Describe the RN (20 points)

Description of RN can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/milestone1/milestone1.ipynb).

**NOTE:** Above description, shape calculation is according to the parameters as mentioned in the [paper](https://arxiv.org/pdf/1706.01427.pdf). The reference repository given in the problem description works with a model with smaller kernels and a very different interpretation of questions embeddings.

## QA on Sort-of-CLEVR (80 points)

### (40 points) Replicate the sort-of-CLEVR dataset result of the paper as quoted in the section 5.3. Please focus only on the CNN, augmented with RN, case.

- With the [github repository](https://github.com/kimhc6028/relational-networks) as reference, we developed the model architecture, keeping the model size smaller but changing the question embedding shape(same as the paper).
- After training the model for 20 epochs, we get the final metrics as follows:

**Relational Data:**

On training set:

| Accuracy(%) | Loss |
|---|---|
| 90.19 | 0.223 |

On testing set:

| Accuracy(%) | Loss |
|---|---|
| 83.22 | 0.386 |

**Non-Relational Data:**

On training set:

| Accuracy(%) | Loss |
|---|---|
| 99.95 | 0.002 |

On testing set:

| Accuracy(%) | Loss |
|---|---|
| 99.85 | 0.003 |

Entire training log can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/logs.txt).

Accuracy and Loss plots can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/tree/main/output):

![Accuracy Plot](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/acc.jpg)

![Loss Plot](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/loss.jpg)


With a larger model, as mentioned in the paper, we will be able to achieve higher accuracy for relational data. But training that larger of a model will require better computational capabilities.

Sample test output can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/tree/main/output):

![Test 1](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/test_0.jpg)

![Test 2](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/test_15.jpg)

![Test 3](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/test_30.jpg)

![Test 4](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/output/test_45.jpg)

### (40 points) Perform the “state description” task and create a table that represents the state of each image.

State descriptors(as a CSV file) for the entire dataset can be found [here](https://github.com/atharva-bhagwat/CSGY-6613/blob/main/sort_of_clevr/sort_of_clevr_descriptor.csv). The CSV file can be loaded as a dataframe for training purposes.

Sample state descriptor for one image:

| image_id | color | center | shape| area | dataset |
|---|---|---|---|---|---|
| 83b5db9f-660e-4658-8d68-3a3889fc557a | red | [56, 19] | circle | 31.41592654 | train |
| 83b5db9f-660e-4658-8d68-3a3889fc557a | green | [28, 7] | rectangle | 100 | train |
| 83b5db9f-660e-4658-8d68-3a3889fc557a | blue | [6, 34] | rectangle | 100 | train |
| 83b5db9f-660e-4658-8d68-3a3889fc557a | orange | [68, 64] | rectangle | 100 | train |
| 83b5db9f-660e-4658-8d68-3a3889fc557a | gray | [53, 63] | rectangle | 100 | train |
| 83b5db9f-660e-4658-8d68-3a3889fc557a | yellow | [68, 7] | circle | 31.41592654 | train |
