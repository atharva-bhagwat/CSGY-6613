import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def translate(data_entry, colors=['red', 'green', 'blue', 'orange', 'gray', 'yellow'], answer_format=['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6']):
  """Helper function to visualize image and it's related questions.

  Args:
  data_entry(tuple): A single row in a dataset
  Format for data_entry:
  image, (rel_questions, rel_answers, rel_pred), (norel_questions, norel_answers, norel_pred) = data_entry
  
  - image: np.array
  - rel_question and norel_question: np.array
  - rel_answers, norel_answer, rel_pred, and norel_pred: int
  """
  image, (rel_question, rel_answer, rel_pred), (norel_question, norel_answer, norel_pred) = data_entry

  image = np.swapaxes(image, 0, 2)
  image = image * 255
  image = cv2.resize(image, (512, 512))

  questions = [rel_question, norel_question]
  answers = [rel_answer, norel_answer]
  pred_answers = [rel_pred, norel_pred]

  for question, answer, pred_answer in zip(questions, answers, pred_answers):
    query = []
    query.append(colors[question.tolist()[0:6].index(1)])
    if question[6] == 1:
      if question[8] == 1:
        query.append('shape?')
      if question[9] == 1:
        query.append('left?')
      if question[10] == 1:
        query.append('up?')
    elif question[7] == 1:
      if question[8] == 1:
        query.append('closest shape?')
      if question[9] == 1:
        query.append('furthest shape?')
      if question[10] == 1:
        query.append('count?')

    answer = answer_format[answer]
    pred_answer = answer_format[pred_answer]
    print(f'Question: {" ".join(query)}\nAnswer: {answer}\nPredicted Answer: {pred_answer}')
    cv2_imshow(image)