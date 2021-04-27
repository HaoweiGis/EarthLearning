import random
import sys
import os 
import argparse



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--work_path', default=r'D:\2_HaoweiPapers\6_MappingForSlum\Datasets\slumSamples', help='work path')
    parser.add_argument('--allnum','-a', default='1091', help='all data num')
    parser.add_argument('--valnum','-v', default='250', help='val data num')
    args = parser.parse_args()
    return args


args = parse_args()
workpath = args.work_path
nums = int(args.allnum)
numval = int(args.valnum)
# nums = int(sys.argv[1])
# num = int(sys.argv[2])
# workpath=sys.argv[3]

allfile = os.path.join(workpath,'all.txt')
trainfile =os.path.join(workpath,'index/train.txt')
testfile =os.path.join(workpath,'index/test.txt')
valfile =os.path.join(workpath,'index/val.txt')

with open(allfile,'r')as f:
    lines = f.readlines()
    g = [i for i in range(1, nums)]# 设置文件总数
    random.shuffle(g)
    # 设置需要的文件数
    train = g[:(nums-numval)]

    for index, line in enumerate(lines,1):
        if index in train:
            with open(trainfile,'a')as trainf:
                trainf.write(line)
        else:
            with open(valfile,'a')as valf:
                valf.write(line)
print(len(train))