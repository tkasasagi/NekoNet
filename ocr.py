image_size = 576
FONTPATH = './resource/NotoSansCJKjp-Regular.otf'
MODELFILE = './model/lowest_loss28.pkl'
from PIL import Image, ImageDraw, ImageFont

from os import listdir
from skimage import io
from skimage.transform import resize
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import torch
from torchvision.utils import save_image
import pandas as pd
from character_set import CharacterSet

font = ImageFont.truetype(FONTPATH, 14, encoding='utf-8')
directory = "./decoder/source/"
resized_directory = './decoder/images/'

filelist = listdir(directory)
'''
resized = (image_size, image_size)

for i in trange(len(filelist)):
    im = io.imread(directory + filelist[i])
    imresized = resize(im, resized)
    io.imsave(resized_directory + filelist[i], imresized)

'''

char = CharacterSet()
model = torch.load(MODELFILE)
model.cuda()

file_list = listdir(resized_directory)

for file in file_list:

    x = []
    y = []
    prob = []
    char2 = []
    result_df = pd.DataFrame()

    image = io.imread(resized_directory + file)
 
    image = np.swapaxes(image, 0, 2) 
    image = np.swapaxes(image, 1, 2)
    image = np.reshape(image, (1, 3, image_size, image_size)) 

    image = torch.from_numpy(image.astype('float32')).cuda() / 255.0

    print(image.min(), image.max())
    
    a,b = model(image)
    a = a.cpu()
    b = b.cpu()

    for i in range(image_size):
        for j in range(image_size):
            max_prob, max_ind = a[:,:,i,j].max(dim=1)
            max_ind = max_ind.item()
            max_prob = max_prob.item()
            max_char = char.ind2char[int(max_ind)]
            
            if max_prob > 0.4 and b[:,:,i,j] > 0.5:
                #print(i, j, max_prob, max_char)
                x.append(j)
                y.append(i)
                prob.append(max_prob)
                char2.append(max_char)
            

    result_df['y'] = y
    result_df['x'] = x
    result_df['prob'] = prob
    result_df['char'] = char2
    result_df.to_csv("./decoder/csv/" + file[0:-4] + '.csv', index = False)
    #result_df.to_csv("./cluster/" + file[0:-4] + '.csv', index = False)

#Clustering ####################################################################


directory = "./decoder/"
image_file = listdir(directory + "images/")



#dictionary file
dictionary = pd.read_csv("./resource/dictionary.csv")

for j in trange(len(image_file)):
    file_name = image_file[j]


    #Open files we need for result
    result = pd.read_csv(directory + "csv/" + file_name[0:-4] + ".csv")
    if (result.shape[0] > 0):
    
        clustering = DBSCAN(eps=3, min_samples=1)
        
        labels = clustering.fit_predict(result[['y', 'x']].values)
        
        pd.Series(labels).value_counts()
        
        result['label'] = labels
        
        xs = []
        ys = []
        chars = []
        for l, group in result.groupby('label'):
        #     print(group)
            x, y = int(group['x'].mean()), int(group['y'].mean())
            char = group['char'].values[0]
            #print(x, y, char)
    
            xs.append(x)
            ys.append(y)
            chars.append(char)
        
        cluster_result = pd.DataFrame()
        cluster_result['char'] = chars
        cluster_result['x'] = xs
        cluster_result['y'] = ys
        cluster_result.to_csv(directory + "cluster/" + file_name[0:-4] + '.csv', index = False)
        
        image = Image.open(directory + "images/" + file_name[0:-4] + ".jpg")
        draw = ImageDraw.Draw(image)
        
        for x, y, char in zip(xs, ys, chars):
            character = dictionary[dictionary['Unicode'] == char].iloc[0,1]
            
            color = 'rgb(255, 0, 0)' 
             
            draw.text((x + 10, y - 10), character, fill=color, font = font)
            
        image.save(directory + "ocr/" + file_name[0:-4] + ".jpg")
