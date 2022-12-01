import os
import cv2
import pickle
import pandas as pd
import numpy as np
from uuid import uuid4
from tqdm import tqdm
np.random.seed(42)

class sortOfClevr():
  def __init__(self, path="", train_size=9800, test_size=200):
    """Initialize parameters for data generator class

    Args:
        path (str): Path to store generated dataset. Defaults to "".
        train_size (int): Size of training set. Defaults to 9800.
        test_size (int): Size of testing set. Defaults to 200.
    """
    self.train_size = train_size
    self.test_size = test_size
    self.image_size = 75  # image size 75 * 75
    self.shape_size = 5   # size of each shape in an image (length of a square/radius of a circle)
    self.question_size = 11 # question embedding length
    self.question_type_idx = 6  # index for question type
    self.question_subtype_idx = 8 # index for question sub-type
    
    self.num_question = 10  # number of questions

    self.color_mapping = {
      0: 'red',
      1: 'green',
      2: 'blue',
      3: 'orange',
      4: 'gray',
      5: 'yellow'
    }
    self.colors = [(0,0,255),
            (0,255,0),
            (255,0,0),
            (0,156,255),
            (128,128,128),
            (0,255,255)]
            
    self.parent_path = path
    self.data_path = os.path.join(self.parent_path, "sort_of_clevr")
    self.filename = os.path.join(self.data_path, "sort_of_clevr.pkl")
    
    self.state_descriptor_filename = os.path.join(self.data_path, "sort_of_clevr_descriptor.csv")
    self.state_descriptor = pd.DataFrame(columns=['image_id','color','center','shape','area','dataset'])

    self.driver() # call driver function for generation
              
  def setup_dirs(self):
    """Helper function to setup directories
    """
    os.makedirs(self.data_path, exist_ok=True)
      
  def get_center(self, objects):
    """Helper function to randomly generate center points for shapes in an image

    Args:
        objects (list): List of generated shapes in the image
    Returns:
        tuple: Center of shape
    """
    while True:
      flag = True
      center = np.random.randint(self.shape_size, self.image_size-self.shape_size, 2)
      for _, c, _ in objects:
        if np.linalg.norm(center - c) < (self.shape_size*3):
          flag = False
      if flag:
        return center

  def add_state_descriptor(self, objects, mode):
    """Function to add an entry to the state descriptors

    Args:
        objects (list): List of shapes and their properties in an image
        mode (str): train/test mode the image belongs to
    """
    image_id = uuid4()  # randomly generate an ID for a image, to save in a data-frame
    for itr_obj in objects:
      shape = 'rectangle' if itr_obj[2] == 'r' else 'circle'
      area = (self.shape_size*2)**2 if shape == 'rectangle' else 2*np.pi*self.shape_size
      self.state_descriptor = self.state_descriptor.append({
        'image_id':image_id,
        'color': self.color_mapping[itr_obj[0]],
        'center': itr_obj[1],
        'shape': shape,
        'area': area,
        'dataset': mode
      }, ignore_index=True)

  def write_state_descriptor(self):
    """Function to save data-frame as a CSV
    """
    self.state_descriptor.to_csv(self.state_descriptor_filename, index=False)
      
  def build_dataset(self, mode):
    """Main function which generates an image, 10 relational questions,
    10 non-relational questions, and answers for all the questions

    Args:
    mode(str): train/test mode the image belongs to
    """
    # generate image with shapes of all colors
    objects = []
    image = np.ones((self.image_size, self.image_size, 3)) * 255
    for color_id, color in enumerate(self.colors):
      center = self.get_center(objects)
      if np.random.random() < 0.5:
        # create a rectangle
        left_top = (center[0]-self.shape_size, center[1]-self.shape_size)
        right_bottom = (center[0]+self.shape_size, center[1]+self.shape_size)
        image = cv2.rectangle(image, left_top, right_bottom, color, -1)
        objects.append((color_id, center, 'r'))
      else:
        # create a circle
        center_ = (center[0], center[1])
        image = cv2.circle(image, center_, self.shape_size, color, -1)
        objects.append((color_id, center, 'c'))

    self.add_state_descriptor(objects, mode)
            
    rel_questions = []
    rel_answers = []
    
    # relational data
    for _ in range(self.num_question):
      question = np.zeros((self.question_size))
      color = np.random.randint(0, 5)
      question[color] = 1
      question[self.question_type_idx+1] = 1
      subtype = np.random.randint(0,2)
      question[subtype+self.question_subtype_idx] = 1
      rel_questions.append(question)
      if subtype == 0:
        # closest to -> 'r'/'c'
        current_center = objects[color][1]
        distances = [np.linalg.norm(current_center - itr_obj[1]) for itr_obj in objects]
        distances[distances.index(0)] = np.inf
        closest = distances.index(min(distances))
        if objects[closest][2] == 'r':
          answer = 2
        else:
          answer = 3
      elif subtype == 1:
        # furthest from -> 'r'/'c'
        current_center = objects[color][1]
        distances = [np.linalg.norm(current_center - itr_obj[1]) for itr_obj in objects]
        closest = distances.index(max(distances))
        if objects[closest][2] == 'r':
          answer = 2
        else:
          answer = 3
      else:
        # count -> 1-6
        current_shape = objects[color][2]
        count = -1
        for itr_obj in objects:
          if itr_obj[2] == current_shape:
            count += 1
        answer = count
      rel_answers.append(answer)
        
    norel_questions = []
    norel_answers = []
    
    # non-relational data
    for _ in range(self.num_question):
      question = np.zeros((self.question_size))
      color = np.random.randint(0, 5)
      question[color] = 1
      question[self.question_type_idx] = 1
      subtype = np.random.randint(0,2)
      question[subtype+self.question_subtype_idx] = 1
      norel_questions.append(question)
      if subtype == 0:
        # query shape -> 'r'/'c'
        if objects[color][2] == 'r':
            answer = 2
        else:
            answer = 3
      elif subtype == 1:
        # query horizontal position -> 'yes'/'no'
        if objects[color][1][0] < self.image_size/2:
          answer = 0
        else:
          answer = 1
      else:
        # query vertical position -> 'yes'/'no'
        if objects[color][1][1] < self.image_size/2:
          answer = 0
        else:
          answer = 1
      norel_answers.append(answer)
    
    norel_data = (norel_questions, norel_answers)
    rel_data = (rel_questions, rel_answers)
    
    image = image/255.
    dataset = (image, rel_data, norel_data)
    return dataset
      
  def driver(self):
    """Driver function which generates training set, testing set, and state descriptors
    """
    self.setup_dirs()

    print('Generating training set...')
    train_dataset = []
    for _ in tqdm(range(self.train_size)):
      train_dataset.append(self.build_dataset(mode='train'))

    test_dataset = []
    print('Generating testing set...')
    for _ in tqdm(range(self.test_size)):
      test_dataset.append(self.build_dataset(mode='test'))

    print('Saving...')
    with open(self.filename, 'wb') as fd:
      pickle.dump((train_dataset, test_dataset), fd)
    print(f'Dataset saved at {self.filename}...')
    
    self.write_state_descriptor()
    print(f'State descriptors saved at {self.state_descriptor_filename}...')
        
if __name__ == "__main__":
    gen = sortOfClevr()