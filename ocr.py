image_size = 576
fontsize = 60
FONTPATH = './resource/NotoSansCJKjp-Regular.otf'

from PIL import Image, ImageDraw, ImageFont
from os import listdir
from skimage import io
from skimage.transform import resize
import numpy as np
from sklearn.cluster import DBSCAN 
import torch
import pandas as pd
from character_set import CharacterSet
import time
import os
from functions import *


font = ImageFont.truetype(FONTPATH, fontsize, encoding='utf-8')
directory = "./test_images/"
resized_directory = './resized_test/'
ocr_folder = './ocr/'
dictionary = pd.read_csv("./resource/dictionary.csv")
filelist = listdir(directory)
# Lookup table for characters for drawing
charmap = {unicode: char for unicode, char in dictionary[['Unicode', 'character']].values}
modelfile = './model/lowest_loss.pkl'
ocrtime = []
#------------- FUNCTION -------------------------------
def fast_ocr(modelname, file):
    t0 = time.time()
    print("Processing file", file)
    t0 = time.time()
    # Model recognition. It probably doesn't get any faster than this.
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
  
    prob_thresh = 0.1

    with torch.no_grad():
      top_ind, top_p, prox_output = model.evaluate(image.data, prob_thresh)

    top_p = top_p.cpu()
    top_ind = top_ind.cpu()
    prox_output = prox_output.cpu()

    prox_pos = (prox_output[0,0] > prob_thresh).data.cpu() & (top_p[0,0] > 0.1)

    #print('prox_pos shape', prox_pos.shape)

    y, x = np.where(prox_pos)

    for i, j in zip(y, x):
        prob.append(top_p[0,0,i,j])
        char2.append(char.ind2char[int(top_ind[0,0,i,j])])

    del top_p, top_ind, prox_output
    #result before clustering. Many characters predicted at a center point.
    result_df['y'] = y
    result_df['x'] = x
    result_df['prob'] = prob
    result_df['char'] = char2
    print('got csv')
    
    print('result df shape', result_df.shape)

    # If the prediction is not zero.
    if (result_df.shape[0] > 0):
        
        #clustering
        clustering = DBSCAN(eps=1.0, min_samples=1) 
       
        labels = clustering.fit_predict(result_df[['y', 'x']].values)
        
        result_df['label'] = labels
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        xs = []
        ys = []
        cha = []
        for l, group in result_df.groupby('label'):
            x, y = int(group['x'].mean()), int(group['y'].mean())
            x1, y1 = int(group['x'].min()), int(group['y'].min())
            x2, y2 = int(group['x'].max()), int(group['y'].max())
            chax = group['char'].values[0]
    
            xs.append(x)
            ys.append(y)
            xmin.append(x1)
            ymin.append(y1)
            xmax.append(x2)
            ymax.append(y2)
            cha.append(chax)
        #clustering result df
        cluster_result = pd.DataFrame()
        cluster_result['char'] = cha
        cluster_result['x'] = xs
        cluster_result['y'] = ys
        print('got clusters')

        #Draw result on image
        imsource = Image.open(directory + 'source/' + file).convert('RGBA')

        width, height = imsource.size
        tmp = Image.new('RGBA', imsource.size)
        
        draw = ImageDraw.Draw(tmp)
        rx = width/image_size
        ry = height/image_size
        
        fx = []
        fy = []
        
        for x, y, chax in zip(xs, ys, cha):
            character = charmap[chax]          
            color = 'rgb(255, 0, 0)'
            xz = x * rx #calculate back to original size
            yz = y * ry #calculate back to original size
            fx.append(xz)
            fy.append(yz)
            xb = xz - fontsize * 0.5 #move coordinate a bit so it won't be right in the middle of character
            yb = yz - fontsize/3      
            draw.rectangle((xb - 3, yb, xb + fontsize , yb + fontsize), fill=(255, 242, 204, 120))
            draw.text((xb, yb - fontsize/3), character, fill=color, font = font)
        imsource = Image.alpha_composite(imsource, tmp)
        imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.
        
        #if random.uniform(0,1) < 0.05:
        imsource.save(ocr_folder + file)
        
        print('OCR image saved')
        final_result = pd.DataFrame()
        final_result['char'] = cha#fchar
        final_result['x'] = fx
        final_result['y'] = fy
        
        #Save Bounding boxes
        boundingbox = pd.DataFrame()
        boundingbox['Unicode'] = cha
        boundingbox['xmin'] = [q * rx for q in xmin]
        boundingbox['xmax'] = [q * rx for q in xmax]
        boundingbox['ymin'] = [q * ry for q in ymin]
        boundingbox['ymax'] = [q * ry for q in ymax]
        boundingbox.to_csv("./boundingbox/" + file[0:-4] + '.csv', index = False)   
        
        #save final csv result.
        final_result.to_csv(final_folder + file[0:-4] + '.csv', index = False)
        time1 = time.time() - t0
        print('OCR time', time1)
        
        return time1
#------------------------------------------------------
filelist = listdir(directory + 'source/')

for model in modelfile:
    
    modelname = model[:-4]
    print(modelname)
    #create final folder
    final_folder = './decoder/' + modelname + '_final/'
    checkdir(final_folder)
    #create ocr folder
        
    resized = (image_size, image_size)
    
    resize_warn = False
    for i in trange(len(filelist)):
        # Only resize files if they are not already resized
        if not os.path.isfile(resized_directory + filelist[i]):
            im = io.imread(directory + 'source/'  + filelist[i])
            imresized = resize(im, resized)
            io.imsave(resized_directory + filelist[i], imresized)
        elif not resize_warn:
            print('Some images are already resized, these are not being resized again')
            resize_warn = True
    
    #-------------------------------------------------------

    char = CharacterSet()
    print('loading from', model)
    model = torch.load('./models/' + model)
    model = model.eval()
    model.cuda()
    model.target_weights = model.compute_target_weights()

    file_list = listdir(resized_directory)

    print('ready to read files')

    for file in file_list:
        ocrtime.append(fast_ocr(modelname, file))
print(sum(ocrtime)/float(len(ocrtime)))
