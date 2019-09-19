from os import listdir
import torch
import pandas as pd

class CharacterSet:

    def __init__(self):       
        import pandas as pd       
        charlist = pd.read_csv('./resource/charlist400.csv')       
        charlist = list(charlist['Unicode'])        
        char2ind = {x: i for i, x in enumerate(charlist)}    
        ind2char = {}

        for char in char2ind:
            ind2char[char2ind[char]] = char

        self.ind2char = ind2char        
        self.char2ind = char2ind
        self.num_characters = len(char2ind)

class KuzushijiDataLoader:

  def __init__(self):
   
    chars = CharacterSet()
    num_char = chars.num_characters
    
    self.num_characters = num_char
    self.num_char = num_char
    self.chars = chars
    
    csvfile = "./input/train.csv"     
    df = pd.read_csv(csvfile)
    obj = {}
    
    for index, row in df.iterrows():
        if isinstance(row['labels'], float):
            continue
        labels = row['labels'].split(' ')
        obj[row['image_id']] = []
        for q in range(len(labels)):
            if q % 5 == 0:
                char = labels[q]
                if char in chars.char2ind:
                    obj[row['image_id']].append([int(labels[q + 1]),int(labels[q + 2]),int(labels[q + 3]),int(labels[q + 4]),chars.char2ind[char]])
                else:
                    obj[row['image_id']].append([int(labels[q + 1]),int(labels[q + 2]),int(labels[q + 3]),int(labels[q + 4]),-1])

    self.obj = obj

  def loadbbox_from_csv(self, target_img, img_size, full_size):
    target_tensor = torch.zeros(size=(img_size[0],self.num_char,img_size[2],img_size[3]))
    prox = torch.zeros(size=(img_size[0],1,img_size[2],img_size[3]))

    if target_img not in self.obj:
      return None, None

    bb_lst = self.obj[target_img]

    xresize = full_size[3]/img_size[3]
    yresize = full_size[2]/img_size[2]

    for bb in bb_lst:
      xpos, ypos, width, height, char_ind = bb
      xpos = int(xpos/xresize)
      width = int(width/xresize)
      ypos = int(ypos/yresize)
      height = int(height/yresize)

      xcenter = xpos + width//2
      ycenter = ypos + height//2

      extra_width = max(width, 10) - width

      if char_ind != -1:
        target_tensor[:,char_ind,ypos:ypos+height,xpos-extra_width//2:xpos+width+extra_width//2] += 1

      prox[:,0,ycenter-2:ycenter+2,xcenter-2:xcenter+2] += 1.0

    prox = torch.clamp(prox, 0.0, 1.0)

    return target_tensor, prox

