import json
import sys
sys.path.append("..")
from config import *

def align( path = '', dataset = 'train' ):
    path_ans = path + 'data/annotations/%s.json'%(dataset)
    path_qus = path + 'data/questions/%s.json'%(dataset)
    path_ans_align = path + 'data/annotations/%s_align.json'%(dataset)
    path_qus_align = path + 'data/questions/%s_align.json'%(dataset)
    
    with open(path_ans,'r') as f_ans:
        dict_ans=json.load(f_ans)
    print('Annotations\' number:', len(dict_ans['annotations']))
    sorted_ans = sorted(dict_ans["annotations"], key=lambda x: (x['question_id']))
    
    with open(path_qus, 'r') as f_qus:
        dict_qus=json.load(f_qus)
    print('Questions\' number:', len(dict_qus['questions']))
    sorted_qus = sorted(dict_qus['questions'], key=lambda x: (x['question_id']))
    
    not_matching_num = 0
    for q, a in list(zip(sorted_qus, sorted_ans)):
        if q['question_id'] != a['question_id']:
            print("Not matching: ", q['question_id'], a['question_id'])
            not_matching_num += 1
    
    if not_matching_num == 0:
        print( "All Q&A match.")
        dict_ans["annotations"] = sorted_ans
        with open(path_ans_align, 'w') as f_ans_align:
            f_ans_align.write(json.dumps(dict_ans))
        dict_qus["questions"] = sorted_qus
        with open(path_qus_align, 'w') as f_qus_align:
            f_qus_align.write(json.dumps(dict_qus))

if __name__ == '__main__':
    path = '../' + path_data
    align(path, 'train')
    align(path, 'val')
    align(path, 'test')
