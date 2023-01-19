from Unsupervised_GreedyHash import GreedyHashModelUnsupervised
from app import model , model_state_dict
import numpy as np
from numpy import save
import mmcv
import torch
import cv2
from torchvision import transforms
import os
from PIL import Image
import glob
import time

video_path='/data/vhngai/Hashing/Videos/'
hash_path='/data/vhngai/Hashing/hashcodes(64)/'
#file = '/data/vhngai/Hashing/test.mp4'

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])
'''
def load_model():
    model = GreedyHashModelUnsupervised(16)
    model.load_state_dict(model_state_dict)
    model.eval()
    print("Model loaded successfully")
    return model
'''

def encoding_video(file, model):
    #name = file.split('.')
    #filename = name[0].split('/')
    #print(filename[-1])

    file_name = os.path.basename(file)
    print(file_name)

    video_reader = mmcv.VideoReader(file)
    npy_list = []
    i = 0
    for frame in mmcv.track_iter_progress(video_reader):
        if i % 6 == 0: # Sample 5 frames in one second, assuming the rate is 30
            frame = Image.fromarray(frame)#Convert to PIL image from numpy array
            f = transform(frame)#change frame into the right size
            f = f.unsqueeze(0)
            #f = np.asarray(f)
            #frame_list.append(f) #add all frames into frame_list
            #put everything below into here to scan each frame then save it into a list then i can save it into the npy file.
            
            #frame_tensor=np.asarray(frame_list)#stack to the 4th dimention
            #frame_tensor = torch.from_numpy(frame_tensor)	
            #frame_tensor = torch.Tensor(frame_tensor)
            #print("Seconds since epoch =", seconds)	
            qB = model(f).sign()[0].detach().numpy()
            #print("Seconds since epoch =", seconds)	
            qB = qB.flatten().tolist()	
            #print(qB)
            npy_list.append(qB)
                    
        i += 1

    
    path = os.path.join(hash_path)
        
    with open(path + file_name + '.npy', 'wb') as f:

        #np.save(path + filename,qB)
        np.save(f, npy_list)
    
    
    print('saved video')

    #with open('hashcode.txt','w') as f: #use a to append and not w
    #    f.write(' \n')
    #    f.write(filename[-1])
    #    f.write(' ')
    #    np.savetxt(f, qB, fmt="%d", delimiter=' ')

        #print(qB.shape)
    #save qB into a npy file.

def video_wrapper(video_path):
    for file in glob.iglob(f'{video_path}/*'):
        #print(file)
        #with open('hashing.txt', 'w') as f:
         #   f.write(file)
        encoding_video(file, model)
        
        


if __name__=='__main__':
    video_wrapper(video_path)
    #encoding_video(file, model)
    

    


'''
    video = []
    for frame in frames:
        f = transform(frame)
        video.append(f)
    video = np.asarray(video)
    qB = model(video).sign()[0].detach().numpy()
    print(qB)
    print(qB.shape)
'''

