image_size = 576
fontsize =30
FONTPATH = './resource/NotoSansCJKjp-Regular.otf'
MODELFILE = './model/lowest_loss.pkl'
#MODELFILE = './model/res_nomix_model.pkl'
from PIL import Image, ImageDraw, ImageFont

net = '_resnet/'
#net = '_nomix/'

import imagesize
from os import listdir
import os
from skimage import io
from skimage.transform import resize
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import torch
from torchvision.utils import save_image
import pandas as pd
from kuzushiji_get_targets import CharacterSet
import shutil

font = ImageFont.truetype(FONTPATH, fontsize, encoding='utf-8')
directory = "./result/"
resized_directory = directory + 'images/'

filelist = listdir(directory + 'source/')

resized = (image_size, image_size)
'''
for i in trange(len(filelist)):
    im = io.imread(directory + 'source/' + filelist[i])
    imresized = resize(im, resized)
    io.imsave(resized_directory + filelist[i], imresized)
'''


dictionary = pd.read_csv("./resource/unicode_translation.csv")
diclist = dictionary['Unicode'].tolist()

char = CharacterSet()
model = torch.load(MODELFILE)
model.cuda()

file_list = listdir(resized_directory)
gt_map = {}

print('ready to read files')

books_parsed = {}

for file in filelist:

    print("Processing file", file)

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

    with torch.no_grad():
        output, prox_output = model(image.data)

    top_p, top_ind = torch.max(output,1,keepdim=True)

    top_p = top_p.cpu()
    top_ind = top_ind.cpu()
    prox_output = prox_output.cpu()

    for i in range(image_size):
        for j in range(image_size):
            #max_prob, max_ind = a[:,:,i,j].max(dim=1)
            
            max_ind = top_ind[:,:,i,j].item()
            max_prob = top_p[:,:,i,j].item()
            prox_pos = prox_output[:,:,i,j].item()
            max_char = char.ind2char[int(max_ind)]
            
            #print('min max ind', top_ind.min(), top_ind.max())
            #print('min max p ', top_p.min(), top_p.max())

            #using 0.001 and 0.8
            if max_prob > 0.001 and prox_pos > 0.6:
                #print("FOUND CHARACTER", i, j, max_prob, max_char)
                x.append(j)
                y.append(i)
                prob.append(max_prob)
                char2.append(max_char)

    del top_p, top_ind, prox_output

    result_df['y'] = y
    result_df['x'] = x
    result_df['prob'] = prob
    result_df['char'] = char2
    print('save to csv')
    result_df.to_csv(directory + "csv/" + file[0:-4] + '.csv', index = False)

    if (result_df.shape[0] > 0):
    
        clustering = DBSCAN(eps=3, min_samples=1)
        
        labels = clustering.fit_predict(result_df[['y', 'x']].values)
        
        pd.Series(labels).value_counts()
        
        result_df['label'] = labels
        
        xs = []
        ys = []
        cha = []
        for l, group in result_df.groupby('label'):
        #     print(group)
            x, y = int(group['x'].mean()), int(group['y'].mean())
            chax = group['char'].values[0]
            #print(x, y, char)
    
            xs.append(x)
            ys.append(y)
            cha.append(chax)
        
        cluster_result = pd.DataFrame()
        cluster_result['char'] = cha
        cluster_result['x'] = xs
        cluster_result['y'] = ys
        cluster_result.to_csv(directory + "cluster" + "/" + file[0:-4] + '.csv', index = False)
        
        #image = Image.open(directory + "images/" + file)
        imsource = Image.open(directory + "source/" + file).convert('RGBA')
        #print(imsource.mode)
        width, height = imsource.size
        tmp = Image.new('RGBA', imsource.size)
        
        #draw = ImageDraw.Draw(image)
        draw = ImageDraw.Draw(tmp)
        rx = width/image_size
        ry = height/image_size
        
        for x, y, chax in zip(xs, ys, cha):
            character = dictionary[dictionary['Unicode'] == chax].iloc[0,1]            
            color = 'rgb(255, 0, 0)'  
            xb = (x * rx) - fontsize * 0.5
            yb = (y * ry) - fontsize/3      
            draw.rectangle((xb - 3, yb, xb + fontsize , yb + fontsize), fill=(255, 242, 204, 120))
            draw.text((xb, yb - fontsize/3), character, fill=color, font = font)
        imsource = Image.alpha_composite(imsource, tmp)
        imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

        imsource.save(directory + "ocr" + "/" + file)
































