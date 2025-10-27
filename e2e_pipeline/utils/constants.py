import numpy as np
import pdb 

bird_part_labels = {
    'head':0, 'torso':1, 'neck':2, 'lwing':3, 'rwing':4, 'lleg':5, 'lfoot':6, 'rleg':7, 'rfoot':8, 'tail':9
}
bird_part_labels_inv = {v:k for k,v in bird_part_labels.items()}

cat_part_labels = {'head':0, 'torso':1, 'neck':2, 'lfleg':3, 'lfpa':4, 'rfleg':5, 'rfpa':6, 'lbleg':7, 'lbpa':8, 'rbleg':9, 'rbpa':10, 'tail':11}
cat_part_labels_inv = {v:k for k,v in cat_part_labels.items()}

cow_part_labels = {'head':0, 'lhorn':1, 'rhorn':2, 'torso':3, 'neck':4, 'lfuleg':5, 'lflleg':6, 'rfuleg':7, 'rflleg':8, 'lbuleg':9, 'lblleg':10, 'rbuleg':11, 'rblleg':12, 'tail':13}
cow_part_labels_inv = {v:k for k,v in cow_part_labels.items()}

sheep_part_labels = cow_part_labels
sheep_part_labels_inv = cow_part_labels_inv 

dog_part_labels = {'head':0, 'torso':1, 'neck':2, 'lfleg':3, 'lfpa':4, 'rfleg':5, 'rfpa':6, 'lbleg':7, 'lbpa':8, 'rbleg':9, 'rbpa':10, 'tail':11, 'muzzle':12}
dog_part_labels_inv = {v:k for k,v in dog_part_labels.items()}

horse_part_labels = {'head':0, 'lfho':1, 'rfho':2, 'torso':3, 'neck':4, 'lfuleg':5, 'lflleg':6, 'rfuleg':7, 'rflleg':8, 'lbuleg':9, 'lblleg':10, 'rbuleg':11, 'rblleg':12, 'tail':13, 'lbho':14, 'rbho':15}
horse_part_labels_inv = {v:k for k,v in horse_part_labels.items()}

person_part_labels = {'head':0, 'torso':1, 'neck': 2, 'llarm': 3, 'luarm': 4, 'lhand': 5, 'rlarm':6, 'ruarm':7, 'rhand': 8, 'llleg': 9, 'luleg':10, 'lfoot':11, 'rlleg':12, 'ruleg':13, 'rfoot':14}
person_part_labels_inv = {v:k for k,v in person_part_labels.items()}

aeroplane_part_labels_inv = {0:'body', 1:'left_wing', 2:'right_wing', \
                        3:'stern', 4:'tail'}
for i in range(0, 6):
   aeroplane_part_labels_inv[5+i] = f'engine'
for i in range(0,7):
   aeroplane_part_labels_inv[11+i] = f"wheel"

aeroplane_part_labels = {v:k for k,v in aeroplane_part_labels_inv.items()}


bicycle_part_labels_inv = {0:'body', 1:'front_wheel', 2:'back_wheel', 3:'chainwheel', 4:'handlebar', 5:'saddle', 6:'headlight'}
bicycle_part_labels = {v:k for k,v in bicycle_part_labels_inv.items()}

# motorbike_part_labels_inv = {0:'body', 1:'front wheel', 2:'back wheel', 3:'saddle', 4:'handlebar', 5:'headlight', 6:'headlight', 7:'headlight'}
motorbike_part_labels_inv = {0:'body', 1:'front wheel', 2:'back wheel', 3:'saddle', 4:'handlebar'}
motorbike_part_labels = {v:k for k,v in motorbike_part_labels_inv.items()}

pottedplant_part_labels_inv = {0:'body', 1:'pot', 2:'plant'}
pottedplant_part_labels  = {v:k for k,v in pottedplant_part_labels_inv.items()}


bird_expanded_labels_map  ={'head': 'head', 'torso': 'torso', 'neck': 'neck', 'lwing': 'left wing', 'rwing': 'right wing', 'lleg': 'left leg',\
                            'lfoot': 'left foot', 'rleg': 'right leg', 'rfoot': 'right foot', 'tail': 'tail'}


cat_expanded_labels_map = {'head': 'head', 'torso': 'torso', 'neck': 'neck', 'lfleg': 'left front leg', \
                           'lfpa': 'left front paw', 'rfleg': 'right front leg', 'rfpa': 'right front paw', 'lbleg': 'left back leg', 'lbpa': 'left back paw', 'rbleg': 'right back leg', 'rbpa': 'right back paw', 'tail': 'tail'}

cow_expanded_labels_map = {'head': 'head', 'lhorn': 'left horn', 'rhorn': 'right horn', 'torso': 'torso', 'neck': 'neck', 'lfuleg': 'left front upper leg', 'lflleg': 'left front lower leg', 'rfuleg': 'right front upper leg', 'rflleg': 'right front lower leg', 'lbuleg': 'left back upper leg', 'lblleg': 'left back lower leg', 'rbuleg': 'right back upper leg', 'rblleg': 'right back lower leg', 'tail': 'tail'}

dog_expanded_labels_map = {'head': 'head', 'torso': 'torso', 'neck': 'neck', 'lfleg': 'left front leg', 'lfpa': 'left front paw', 'rfleg': 'right front leg', 'rfpa': 'right front paw', 'lbleg': 'left back leg', 'lbpa': 'left back paw', 'rbleg': 'right back leg', 'rbpa': 'right back paw', 'tail': 'tail', 'muzzle': 'muzzle'}


horse_expanded_labels_map = {'head': 'head', 'lfho': 'left front hoof', 'rfho': 'right front hoof', 'torso': 'torso', 'neck': 'neck', 'lfuleg': 'left front upper leg', 'lflleg': 'left front lower leg', 'rfuleg': 'right front upper leg', 'rflleg': 'right front lower leg', 'lbuleg': 'left back upper leg', 'lblleg': 'left back lower leg', 'rbuleg': 'right back upper leg', 'rblleg': 'right back lower leg', 'tail': 'tail', 'lbho': 'left back hoof', 'rbho': 'right back hoof'}

person_expanded_labels_map = {'head': 'head', 'torso': 'torso', 'neck': 'neck', 'llarm': 'left lower arm', 'luarm': 'left upper arm', 'lhand': 'left hand', 'rlarm': 'right lower arm', 'ruarm': 'right upper arm', 'rhand': 'right hand', 'llleg': 'left lower leg', 'luleg': 'left upper leg', 'lfoot': 'left foot', 'rlleg': 'right lower leg', 'ruleg': 'right upper leg', 'rfoot': 'right foot'}

ALL_EXPANDED_LABELS_MAPPING = {
    "bird": bird_expanded_labels_map, 
    "cat": cat_expanded_labels_map, 
    "cow": cow_expanded_labels_map, 
    "sheep": cow_expanded_labels_map, 
    "dog": dog_expanded_labels_map, 
    "horse": horse_expanded_labels_map, 
    "person": person_expanded_labels_map
}

ALL_PART_MAPPING = {
    "bird" : bird_part_labels,
    "cat": cat_part_labels,
    "cow": cow_part_labels,
    "sheep": cow_part_labels,
    "dog": dog_part_labels,
    "horse": horse_part_labels,
    "person": person_part_labels,
    "aeroplane": aeroplane_part_labels, 
    "bicycle": bicycle_part_labels, 
    "motorbike": motorbike_part_labels, 
}

ALL_PART_MAPPING_INV = {
    "bird" : bird_part_labels_inv,
    "cat": cat_part_labels_inv,
    "cow": cow_part_labels_inv,
    "sheep": cow_part_labels_inv,
    "dog": dog_part_labels_inv,
    "horse": horse_part_labels_inv,
    "person": person_part_labels_inv,
    "aeroplane": aeroplane_part_labels_inv, 
    "bicycle":bicycle_part_labels_inv, 
    "motorbike": motorbike_part_labels_inv
}

bird_pose = {}
bird_pose["Left_pose"] = ["head", "neck", "torso", "lwing", "lleg", "lfoot", "rleg", "rfoot", "tail"]
bird_pose["Right_pose"] = ["head", "neck", "torso", "rwing", "rleg", "rfoot", "lleg", "lfoot", "tail"]
dog_pose = {}
# dog_pose["Left_pose"] =  ["head", "neck", "torso", "lfleg", "lfpa", "lbleg", "lbpa", "rbleg", "rbpa", "tail"]
# dog_pose["Right_pose"] = 
# 'head':0, 'torso':1, 'neck':2, 'lwing':3, 'rwing':4, 'lleg':5, 'lfoot':6, 'rleg':7, 'rfoot':8, 'tail':9

ALL_PART_POSE = {
    "bird": bird_pose, 
    "dog":dog_pose
}


bird_labels = {
    'head':1, 'leye':2, 'reye':3, 'beak':4, 'torso':5, 'neck':6, 'lwing':7, 'rwing':8, 'lleg':9, 'lfoot':10, 'rleg':11, 'rfoot':12, 'tail':13
}

colors = [(1, 0, 0),
          (0.737, 0.561, 0.561),
          (0.255, 0.412, 0.882),
          (0.545, 0.271, 0.0745),
          (0.98, 0.502, 0.447),
          (0.98, 0.643, 0.376),
          (0.18, 0.545, 0.341),
          (0.502, 0, 0.502),
          (0.627, 0.322, 0.176),
          (0.753, 0.753, 0.753),
          (0.529, 0.808, 0.922),
          (0.416, 0.353, 0.804),
          (0.439, 0.502, 0.565),
          (0.784, 0.302, 0.565),
          (0.867, 0.627, 0.867),
          (0, 1, 0.498),
          (0.275, 0.51, 0.706),
          (0.824, 0.706, 0.549),
          (0, 0.502, 0.502),
          (0.847, 0.749, 0.847),
          (1, 0.388, 0.278),
          (0.251, 0.878, 0.816),
          (0.933, 0.51, 0.933),
          (0.961, 0.871, 0.702)]

colors = (np.asarray(colors)*255)

def rgb2hex(color):
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

bird_labels_color = {k:rgb2hex(colors[i]) for i,k in enumerate(bird_part_labels.keys())}
bird_labels_color_rgb = {k:colors[i] for i,k in enumerate(bird_part_labels.keys())}
print(f"{bird_labels_color_rgb = }")
cat_labels_color = {k:rgb2hex(colors[i]) for i,k in enumerate(cat_part_labels.keys())}
cat_labels_color_rgb = {k:colors[i] for i,k in enumerate(cat_part_labels.keys())}
# print(f"{cat_labels_color = }")
print(f"{cat_labels_color_rgb = }")
horse_labels_color = {k:rgb2hex(colors[i]) for i,k in enumerate(horse_part_labels.keys())}
horse_labels_color_rgb = {k:colors[i] for i,k in enumerate(horse_part_labels.keys())}
# print(f"{horse_labels_color = }")
print(f"{horse_labels_color_rgb = }")
cow_labels_color = {k:rgb2hex(colors[i]) for i,k in enumerate(cow_part_labels.keys())}
cow_labels_color_rgb = {k:colors[i] for i,k in enumerate(cow_part_labels.keys())}
# print(f"{cow_labels_color = }")
print(f"{cow_labels_color_rgb = }")
# aeroplane_labels_color = {k:rgb2hex(colors[i]) for i,k in enumerate(aeroplane_part_labels.keys())}
# print(f"{aeroplane_labels_color = }")
person_labels_color = {k:rgb2hex(colors[i]) for i,k in enumerate(person_part_labels.keys())}
person_labels_color_rgb = {k:colors[i] for i,k in enumerate(person_part_labels.keys())}
print(f"{person_labels_color_rgb = }")

dog_labels_color = {k:rgb2hex(colors[i]) for i,k in enumerate(dog_part_labels.keys())}
dog_labels_color_rgb = {k:colors[i] for i,k in enumerate(dog_part_labels.keys())}
print(f"{dog_labels_color_rgb = }")
canvas_size = 512

cat_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'nose':6, 'torso':7, 'neck':8, 'lfleg':9, 'lfpa':10, 'rfleg':11, 'rfpa':12, 'lbleg':13, 'lbpa':14, 'rbleg':15, 'rbpa':16, 'tail':17}

cow_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'muzzle':6, 'lhorn':7, 'rhorn':8, 'torso':9, 'neck':10, 'lfuleg':11, 'lflleg':12, 'rfuleg':13, 'rflleg':14, 'lbuleg':15, 'lblleg':16, 'rbuleg':17, 'rblleg':18, 'tail':19}

dog_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'nose':6, 'torso':7, 'neck':8, 'lfleg':9, 'lfpa':10, 'rfleg':11, 'rfpa':12, 'lbleg':13, 'lbpa':14, 'rbleg':15, 'rbpa':16, 'tail':17, 'muzzle':18}

horse_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'muzzle':6, 'lfho':7, 'rfho':8, 'torso':9, 'neck':10, 'lfuleg':11, 'lflleg':12, 'rfuleg':13, 'rflleg':14, 'lbuleg':15, 'lblleg':16, 'rbuleg':17, 'rblleg':18, 'tail':19, 'lbho':20, 'rbho':21}

person_labels = {'head':1, 'leye':2,  'reye':3, 'lear':4, 'rear':5, 'lebrow':6, 'rebrow':7,  'nose':8,  'mouth':9,  'hair':10, 'torso':11, 'neck': 12, 'llarm': 13, 'luarm': 14, 'lhand': 15, 'rlarm':16, 'ruarm':17, 'rhand': 18, 'llleg': 19, 'luleg':20, 'lfoot':21, 'rlleg':22, 'ruleg':23, 'rfoot':24}

condensed_mapping = {
    'cow' : [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    'bird' : [0, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'person' : [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    'cat' : [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'dog' : [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    'horse' : [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
}

person_tree = {
    0: [1, 2, 3, 4, 7, 8, 9, 10, 11],
    1: [0, 2, 3, 5, 7],
    2: [0, 1, 4, 6, 7],
    3: [0, 1],
    4: [0, 2],
    5: [1],
    6: [2],
    7: [0, 1, 2, 8],
    8: [0, 7],
    9: [0],
    10: [0, 11, 13, 16, 19, 22],
    11: [0, 10],
    12: [13, 14],
    13: [10, 12],
    14: [12],
    15: [16, 17],
    16: [10, 15],
    17: [15],
    18: [19, 20],
    19: [18, 10],
    20: [18],
    21: [22, 23],
    22: [21, 10],
    23: [21]}


bird_tree = {
    0: [1, 2, 3, 4, 5],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [0, 1, 2],
    4: [0, 5, 6, 7, 8, 10, 12],
    5: [0, 4],
    6: [7, 4],
    7: [6, 4],
    8: [4, 9],
    9: [8],
    10: [11, 4],
    11: [10],
    12: [4]
}

dog_tree = {
    0: [1, 2, 3, 4, 5, 6, 7, 17],
    1: [2, 0, 3],
    2: [0, 1, 4],
    3: [0, 1],
    4: [0, 2],
    5: [0, 1, 2],
    6: [0, 8, 10, 12, 14, 16, 7],
    7: [0, 6],
    8: [9, 6],
    9: [8],
    10: [6, 11],
    11: [10],
    12: [13, 6],
    13: [12],
    14: [6, 15],
    15: [14],
    16: [6], 17: [0]
}

cat_tree = {
    0: [1, 2, 3, 4, 5, 6, 7],
    1: [2, 0, 3],
    2: [0, 1, 4],
    3: [0, 1],
    4: [0, 2],
    5: [0, 1, 2],
    6: [0, 8, 10, 12, 14, 16, 7],
    7: [0, 6], 8: [9, 6],
    9: [8],
    10: [6, 11],
    11: [10],
    12: [13, 6],
    13: [12],
    14: [6, 15],
    15: [14], 16: [6]
}

horse_tree = {
    0: [1, 2, 3, 4, 5, 8, 9],
    1: [2, 0, 3],
    2: [0, 1, 4],
    3: [0, 1],
    4: [0, 2],
    5: [0, 1, 2],
    6: [11],
    7: [13],
    8: [0, 10, 12, 14, 16, 18],
    9: [0, 8],
    10: [8, 11, 12],
    11: [10, 6],
    12: [10, 8, 13],
    13: [7],
    14: [8, 15, 16],
    15: [14, 19],
    16: [14, 17],
    17: [16, 20],
    18: [8],
    19: [15],
    20: [17]
}

cow_tree = {
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    1: [2, 0, 3, 5],
    2: [0, 1, 4, 5],
    3: [0, 1, 6],
    4: [0, 2, 7],
    5: [0, 1, 2],
    6: [0, 3],
    7: [0, 4, 13],
    8: [0, 9, 10, 12, 14, 16, 18],
    9: [0, 8],
    10: [8, 11, 12],
    11: [10, 6],
    12: [10, 8, 13],
    13: [7, 12],
    14: [8, 15, 16],
    15: [14],
    16: [8, 14, 17],
    17: [16],
    18: [8]
}

ALL_ADJ_MAPPING = {
    "bird" : bird_tree,
    "cat": cat_tree,
    "cow": cow_tree,
    "sheep": cow_tree,
    "dog": dog_tree,
    "horse": horse_tree,
    "person": person_tree,
}

class_dict = {
    0: 'cow',
    1: 'sheep',
    2: 'bird',
    3: 'person',
    4: 'cat',
    5: 'dog',
    6: 'horse',
    7: 'aeroplane',
    8: 'motorbike',
    9: 'bicycle',
    10: 'pottedplant',
}

class_dict_inv = {v:k for k,v in class_dict.items()}



adj_mat_consol = {}
adj_mat_bird = np.array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],dtype=np.float32)

adj_mat_cat = np.array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],dtype=np.float32)

adj_mat_cow = np.array([[1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.],
                        [1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],dtype=np.float32)

adj_mat_dog = np.array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],dtype=np.float32)

adj_mat_horse = np.array([[1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.],
                       [1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.]], dtype=np.float32)

adj_mat_person = np.array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
                           [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
                           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],dtype=np.float32)



adj_mat_consol["bird"] = adj_mat_bird
adj_mat_consol["cat"] = adj_mat_cat
adj_mat_consol["cow"] = adj_mat_cow
adj_mat_consol["sheep"] = adj_mat_cow
adj_mat_consol["dog"] = adj_mat_dog
adj_mat_consol["horse"] = adj_mat_horse
adj_mat_consol["person"] = adj_mat_person





################################################################################################################################
subspecies = {"bird": ["kingfisher", "Macaw", "Blue Jay", "Owl", "Parrot"], \
              "horse": ["Mustang", "Brumby", "Thoroughbred", "Friesian", "Haflinger"], \
              "dog": ["Labrador", "Golden Retriever", "Spaniel", "Rottweiler", "German Shepherd", "Boxer"]}