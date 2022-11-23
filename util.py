import cv2
from google.colab.patches import cv2_imshow

def translate(data_entry, colors=['red', 'green', 'blue', 'orange', 'gray', 'yellow'], answer_format=['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6']):
  """Helper function to visualize image and it's related questions.

  Args:
  data_entry(tuple): A single row in a dataset
  """
  image, (rel_questions, rel_answers), (norel_questions, norel_answers) = data_entry
  questions =  rel_questions + norel_questions
  answers = rel_answers + norel_answers
  
  for question, answer in zip(questions, answers):
    query = ''
    query += colors[question.tolist()[0:6].index(1)]
    if question[6] == 1:
      if question[8] == 1:
        query += 'shape?'
      if question[9] == 1:
        query += 'left?'
      if question[10] == 1:
        query += 'up?'
    elif question[7] == 1:
      if question[8] == 1:
        query += 'closest shape?'
      if question[9] == 1:
        query += 'furthest shape?'
      if question[10] == 1:
        query += 'count?'

    answer = answer_format[answer]
    print(f'Question: {query}\nAnswer: {answer}')
    cv2_imshow(cv2.resize(image,(512,512)))