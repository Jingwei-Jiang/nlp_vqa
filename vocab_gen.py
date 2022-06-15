import json
import dataset
import utils
import itertools
from collections import Counter

def ans_vocab_gen():
    ans_path, ques_path, image_path = utils.path_gen(train=True)

    with open(ans_path, 'r') as fd:
        answers = json.load(fd)

    answer_lists = dataset.prepare_answers(answers)

    all_tokens = itertools.chain.from_iterable(answer_lists)
    counter = Counter(all_tokens)
    ans = counter.keys()
    # 先按个数多少排序，再按字典序排序
    tokens = sorted(ans, key=lambda x: (counter[x], x), reverse=True)
    ans_to_idx = {t: i for i, t in enumerate(tokens)}
    idx_to_ans = {i: t for i, t in enumerate(tokens)}
    print(ans_to_idx)
    print(idx_to_ans)
    return ans_to_idx, idx_to_ans


if __name__ == '__main__':
    ans_to_idx, idx_to_ans = ans_vocab_gen()
    print("Answers' vocabulary size: ", len(ans_to_idx))

    print("Top 5 answers:")
    for (word, index) in ans_to_idx.items():
        print("\'{}\'\tid: {}".format(word, index))
        if index > 3: break;