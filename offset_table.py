from fileinput import filename
import numpy as np
import os
import glob

hash_path="C:\\Users\\Vincent\\OneDrive - UTS\\5 Year\\Honors Project\\Hashing\\hashcodes(64)\\"
save_path="C:\\Users\\Vincent\\OneDrive - UTS\\5 Year\\Honors Project\\Hashing\\"
#test_file = "C:\\Users\\vince\\OneDrive - UTS\\5 Year\\Honors Project\\Hashing\\hashcodes\\MEN IN BLACK (1997) - Official Movie Teaser Trailer (1080p_24fps_H264-128kbit_AAC).mp4.npy"



#vstack all the encodes into a txt file
    
def video_wrapper(hash_path):
    frames = 0
    npy_list = []
    path = os.path.join(save_path)
    for file in glob.iglob(f'{hash_path}/*'):
        file_name = os.path.basename(file)
        print(file_name)
        x = np.load(hash_path + file_name, allow_pickle=True)
        npy_list = np.vstack(x)


        with open(path + 'database64.txt', 'a') as f:
            np.savetxt(f, npy_list, fmt='%d')
        y = x.shape[0]
        frames = frames + y
        
        l = len(file_name)
        name = file_name[:l-8]

        with open(path + 'filename64.txt', 'a') as f:
            f.write(name)
            f.write('\n')
        with open(path + 'frames64.txt', 'a') as f:
            f.write('%01d' % y)
            f.write('\n')
        with open(path + 'offset_table64.txt', 'a') as f:
            f.write('%01d' % frames)
            f.write('\n')
         

    
if __name__=='__main__':
    video_wrapper(hash_path)
    

