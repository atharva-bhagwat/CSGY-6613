import os
import cv2
import numpy as np

def format_question(question, colors=['red', 'green', 'blue', 'orange', 'gray', 'yellow']):
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
    return " ".join(query)

def translate(data_entry, filename, answer_format=['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6'], font=cv2.FONT_HERSHEY_SIMPLEX, font_size=0.3, text_color=(0,0,0), font_thickness=1, font_line=cv2.LINE_AA):
  """Helper function to visualize image and it's related questions.

  Args:
  data_entry(tuple): A single row in a dataset
  Format for data_entry:
  image, (rel_questions, rel_answers, rel_pred), (norel_questions, norel_answers, norel_pred) = data_entry
  
  - image: np.array
  - rel_question and norel_question: np.array
  - rel_answers, norel_answer, rel_pred, and norel_pred: int
  """
  textarea = np.ones((100, 512, 3), dtype=np.uint8) * 255
  
  image, (rel_question, rel_answer, rel_pred), (norel_question, norel_answer, norel_pred) = data_entry

  image = np.swapaxes(image, 0, 2)
  image = image * 255
  image = cv2.resize(image, (512, 512))
  
  image = np.vstack((textarea, image))
  
  # relational question
  rel_ques = format_question(rel_question)
  rel_answer = answer_format[rel_answer]
  rel_pred = answer_format[rel_pred]
  
  image = cv2.putText(image, f"Relational Question: {rel_ques} | Answer: {rel_answer} | Predicted: {rel_pred}", (20,20), font, font_size, text_color, font_thickness, font_line)
  
  # non-relational question
  norel_ques = format_question(norel_question)
  norel_answer = answer_format[norel_answer]
  norel_pred = answer_format[norel_pred]
  
  image = cv2.putText(image, f"Non-Relational Question: {norel_ques} | Answer: {norel_answer} | Predicted: {norel_pred}", (20,40), font, font_size, text_color, font_thickness, font_line)
  
  cv2.imwrite(os.path.join("output", filename), image)
  print(f'{os.path.join("output", filename)} saved...')