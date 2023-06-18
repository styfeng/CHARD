from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from spice.spice import Spice
import spacy
import sys
import codecs
import argparse
from collections import OrderedDict
import json

parser = argparse.ArgumentParser()
parser.add_argument('--key_file', default="", type=str)
parser.add_argument('--gts_file', default="", type=str)
parser.add_argument('--res_file', default="", type=str)
args = parser.parse_args()

nlp = spacy.load("en_core_web_sm")
nlp.pipeline = [('tagger', nlp.tagger)]


def tokenize(dict):
    for key in dict:
        new_sentence_list = []
        for sentence in dict[key]:
            a = ''
            for token in nlp(unicode(sentence)):
                a += token.text
                a += ' '
            new_sentence_list.append(a.rstrip())
        dict[key] = new_sentence_list

    return dict


def evaluator(gts, res, counts):
    eval = {}
    # =================================================
    # Set up scorers
    # =================================================
    print 'tokenization...'
    # Todo: use Spacy for tokenization
    gts = tokenize(gts)
    res = tokenize(res)

    # =================================================
    # Set up scorers
    # =================================================
    print 'setting up scorers...'
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
    ]

    # =================================================
    # Compute scores
    # =================================================
    #names = ["BLEU1", "BLEU2", "BLEU3", "BLEU4"]
    for scorer, method in scorers:
        print 'computing %s score...' % (scorer.method())
        score, scores = scorer.compute_score(gts, res)
        print("len(list(gts.keys())): ",len(list(gts.keys())))
        print("len(scores): ",len(scores))
        if type(method) == list:
            BLEU_dict = {}
            out_file = res_file + '_BLEU.json'
            for sc, scs, m in zip(score, scores, method):
                eval[m] = sc
                unsorted_scores = []
                for score,count in zip(list(scs), counts.values()):
                    unsorted_scores.extend([score])
                BLEU_dict[m] = unsorted_scores
                print "%s: %0.3f" % (m, sc)
            with open(out_file, 'w') as f:
                json.dump(BLEU_dict,f,indent=4)
            f.close()
        else:
            eval[method] = score
            # individual_scores = np.array([sc['All']['f'] for sc in scores])
            unsorted_scores = []
            print("len(list(gts.keys())): ",len(list(gts.keys())))
            print("len(scores): ",len(scores))
            for sc,count in zip(list(scores), counts.values()):
                #unsorted_scores.extend([sc]*count)
                unsorted_scores.extend([sc])
            # print(len(scores), len(unsorted_scores))
            # print([round(sc['All']['f'],4) for sc in scores[:10]])
            # print(np.round(unsorted_scores[:10],4))
            #CIDER_dict["CIDER"] = unsorted_scores
            #out_file = res_file + '_BLEU_list'
            #with open(out_file, 'w') as f:
            #    #json.dump(CIDER_dict,f,indent=4)
            #    print("writing lines to file")
            #    f.writelines('\n'.join([str(x) for x in unsorted_scores]))
            #f.close()
            print("%s: %0.3f" % (method, score))

if __name__=='__main__':

    key_file = args.key_file
    gts_file = args.gts_file
    res_file = args.res_file

    gts = OrderedDict()
    res = OrderedDict()
    counts = OrderedDict()

    with codecs.open(key_file, encoding='utf-8') as f:
        key_lines = f.readlines()
        print("len(key_lines): ",len(key_lines))
        # key_lines = [line.decode('utf-8') for line in f.readlines()]
    with codecs.open(gts_file, encoding='utf-8') as f:
        gts_lines = f.readlines()
        print("len(gts_lines): ",len(gts_lines))
        # gts_lines = [line.decode('utf-8') for line in f.readlines()]
    with codecs.open(res_file, encoding='utf-8') as f:
        res_lines = f.readlines()
        print("len(res_lines): ",len(res_lines))
        # res_lines = [line.decode('utf-8') for line in f.readlines()]

    for key_line, gts_line, res_line in zip(key_lines, gts_lines, res_lines):
        #key = '#'.join(key_line.rstrip('\n').split(' '))
        key = '#'.join(key_line.strip().split(' '))
        if key not in gts:
            gts[key] = []
            gts[key].append(gts_line.rstrip('\n'))
            res[key] = []
            res[key].append(res_line.rstrip('\n'))
            counts[key] = 1
        else:
            gts[key].append(gts_line.rstrip('\n'))
            counts[key] += 1

    evaluator(gts, res, counts)

    # gts = {"cat#dog#boy": ["The dog is the boy's cat.", "The dog eats the cat of the boy."],
    #        "apple#tree#boy": ["A boy is picking apples from trees."]}
    # res = {"cat#dog#boy": ["The dog is the boy's cat."],
    #        "apple#tree#boy": ["A boy is picking apples from trees and put them into bags."]}
    # evaluator(gts, res)
    #
    # gts = {"cat#dog#boy": ["The dog is the boy's cat.", "The dog eats the cat of the boy."],
    #        "apple#tree#boy": ["A boy is picking apples from trees."]}
    # res = {"cat#dog#boy": ["The dog is the boy's cat."],
    #        "apple#tree#boy": ["A boy is picking apples trees and put them into bags"]}
    # evaluator(gts, res)
