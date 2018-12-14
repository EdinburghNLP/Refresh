
# Credits
# Written by Shashi Narayan to use original ROUGE
# Improved by Yang Liu to use a must faster ROUGE


import os
import re
import itertools
from multiprocessing import Pool
import sys
from preprocess import rouge
import codecs


def cal_rouge(fullset, sentdata, golddata):
    fullset.sort()
    model_highlights = [sentdata[idx] for idx in range(len(sentdata)) if idx in fullset]
    rouge_1 = rouge.rouge_n(model_highlights, golddata, 1)['f']
    rouge_2 = rouge.rouge_n(model_highlights, golddata, 2)['f']
    rouge_l = rouge.rouge_l_summary_level(model_highlights, golddata)['f']
    rouge_score = (rouge_1 + rouge_2 + rouge_l)/3.0
    return (rouge_score, fullset)


def _multi_run_wrapper(args):
    return cal_rouge(*args)


def get_fileids(topdir, newstype, datatype, server):
    if newstype == "cnn":
        return open(topdir + "/Temp-ServerFileIds/cnn-" + datatype + "-fileids.txt." + server).read().strip().split(
            "\n")

    if newstype == "dailymail":
        return open(
            topdir + "/Temp-ServerFileIds/dailymail-" + datatype + "-fileids.txt." + server).read().strip().split("\n")

    print "Error: Only CNN and DailyMail allowed."
    exit(0)


if __name__ == "__main__":

    args = sys.argv[1:]

    pool = Pool(int(args[0]))
    data_dir = args[1]
    task = int(args[2])
    sent_limit = 4

    fullcount = 0

    mainbody_dir = os.path.join(data_dir, 'article')
    highlights_dir = os.path.join(data_dir, 'abstracts')

    _listfiles = os.listdir(mainbody_dir)
    all_l = len(_listfiles)
    _listfiles = sorted(_listfiles)
    if (task == 0):
        listfiles = _listfiles[:int(all_l / 3)]
    elif (task == 1):
        listfiles = _listfiles[int(all_l / 3):int(all_l * 2 / 3)]
    elif (task == 2):
        listfiles = _listfiles[int(all_l * 2 / 3):]

    fianllabeldir = os.path.join(data_dir, 'article-oracles')
    if (not os.path.exists(fianllabeldir)):
        os.mkdir(fianllabeldir)

    for summaryfname in listfiles:
        if not re.match('^\d', summaryfname):
            continue
        summaryfname = summaryfname.split('.')[0]
        if os.path.isfile(os.path.join(fianllabeldir, summaryfname + ".f-sent")) and os.path.isfile(
                os.path.join(fianllabeldir, summaryfname + ".moracle")):
            continue
        print(summaryfname)
        sentfull = os.path.join(mainbody_dir, summaryfname + ".article")
        sentdata = (codecs.open(sentfull, encoding='utf-8').readlines())  # full doc
        goldfull = os.path.join(highlights_dir, summaryfname + ".abstracts")
        golddata = (codecs.open(goldfull, encoding='utf-8').readlines())  # full doc

        rougesentwisefile = os.path.join(fianllabeldir, summaryfname + ".f-sent")

        sentids_lst = [[sentid] for sentid in range(len(sentdata))]
        rougescore_sentwise = []
        for sentids in sentids_lst:
            rougescore_sentwise.append(cal_rouge(sentids, sentdata, golddata))

        # print rougescore_sentwise
        foutput = open(rougesentwisefile, "w")
        foutput.write("\n".join([str(item[0]) + "\t" + str(item[1][0]) for item in rougescore_sentwise]) + "\n")
        foutput.close()
        rougescore_sentwise.sort(reverse=True)

        toprougesentences = [item[1][0] for item in rougescore_sentwise[:10]]
        toprougesentences.sort()

        labelfullmodif = fianllabeldir + "/" + summaryfname + ".moracle"
        rougescore_sentids = []
        rougescore_sentids += rougescore_sentwise[:10][:]

        arguments_list = []
        for itemcount in range(2, sent_limit + 1):
            arguments_list += [(list(sentids), sentdata, golddata) for sentids in
                               itertools.combinations(toprougesentences, itemcount)]
        rougescore_sentids = pool.map(_multi_run_wrapper, arguments_list)

        # Process results
        # for rougescore, arguments in zip(rougescore_list, arguments_list):
        #     rougescore_sentids.append((rougescore, arguments[0]))

        # rougescore_sentids = []
        # for sentids in sentids_lst:
        #     rougescore_sentids.append(cal_rouge(sentids, sentdata, golddata))

        rougescore_sentids.sort(reverse=True)

        foutput = open(labelfullmodif, "w")
        for item in rougescore_sentids:
            foutput.write((" ".join([str(sentidx) for sentidx in item[1]])) + "\t" + str(item[0]) + "\n")
        foutput.close()

        if fullcount % 100 == 0:
            print fullcount
        fullcount += 1
