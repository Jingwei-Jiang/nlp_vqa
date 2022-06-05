import os
import json
import sys
sys.path.append("..")
from config import *


def qai_match( path, dataset = 'train' ):
    label_img = 'train' if dataset == 'train' else 'val'
    path_ans = path + 'data/annotations/%s_align.json'%(dataset)
    path_qus = path + 'data/questions/%s_align.json'%(dataset)
    path_img = path + 'data/images/%s/COCO_%s2014_'%( dataset,label_img)
    
    with open(path_ans,'r') as f_ans:
        dict_ans=json.load(f_ans)
    print( dataset + ':Original annotations\' number:', len(dict_ans['annotations']))
    match = filter( lambda x: os.path.isfile(path_img + str(x['image_id']).zfill(12) + '.jpg'), dict_ans['annotations'])
    dict_ans['annotations'] = list(match)
    print(dataset + ': Matched annotations\' number:', len(dict_ans['annotations']))
    
    with open(path_qus,'r') as f_qus:
        dict_qus=json.load(f_qus)
    print(dataset +':Original questions\' number:', len(dict_qus['questions']))
    match = filter( lambda x: os.path.isfile(path_img + str(x['image_id']).zfill(12) + '.jpg'), dict_qus['questions'])
    dict_qus['questions'] = list(match)
    print(dataset +':Matched questions\' number:', len(dict_qus['questions']))
    
    print( "All QA&img match.")
    with open(path_ans, 'w') as f_ans:
        f_ans.write(json.dumps(dict_ans))
    with open(path_qus, 'w') as f_qus:
        f_qus.write(json.dumps(dict_qus))
    
if __name__ == '__main__':
    path = '../' + path_data
    qai_match(path, 'train')
    qai_match(path, 'val')
    qai_match(path, 'test')   