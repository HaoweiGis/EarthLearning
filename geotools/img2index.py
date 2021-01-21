import random
import sys
import os 
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--imgpath','-a', default=r'D:\EarthLearning\preprocessing\datasets\landcover\tile\image', help='all data num')
    parser.add_argument('--outpath','-v', default=r'D:\EarthLearning\preprocessing\datasets\landcover\tile\index', help='val data num',)
    parser.add_argument('--datatype', default=r'tif', help='val data num',)
    args = parser.parse_args()
    return args

def collect_files(dir,suffix,basename=False):
    files = []
    filen = os.path.join(dir, '*.' + suffix)
    filenames = glob.glob(filen)
    for filename in filenames:
        assert filename.endswith(suffix), filename
        if basename:
            filename = os.path.basename(filename).split('.')[0]
            files.append(filename)
        else:
            files.append(filename)
    assert len(files), f'No images found in {dir}'
    print(f'Loaded {len(files)} images from {dir}')
    return files

def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def writefile(filename,lines):
    f = open(filename,'a')
    for line in lines:
        f.writelines(line + '\n')



if __name__ == "__main__":
    args = parse_args()
    basenames = collect_files(args.imgpath,args.datatype,basename=True)
    # print(basenames)
    sublist_1, sublist_2 = data_split(basenames, 0.9)
    writefile(os.path.join(args.outpath, 'train.txt'),sublist_1)
    writefile(os.path.join(args.outpath, 'val.txt'),sublist_2)




# args = parse_args()
# workpath = args.work_path
# nums = int(args.allnum)
# numval = int(args.valnum)
# # nums = int(sys.argv[1])
# # num = int(sys.argv[2])
# # workpath=sys.argv[3]

# allfile = os.path.join(workpath,'all.txt')
# trainfile =os.path.join(workpath,'index/train.txt')
# testfile =os.path.join(workpath,'index/test.txt')
# valfile =os.path.join(workpath,'index/val.txt')

# with open(allfile,'r')as f:
#     lines = f.readlines()
#     g = [i for i in range(1, nums)]# 设置文件总数
#     random.shuffle(g)
#     # 设置需要的文件数
#     train = g[:(nums-numval)]

#     for index, line in enumerate(lines,1):
#         if index in train:
#             with open(trainfile,'a')as trainf:
#                 trainf.write(line)
#         else:
#             with open(valfile,'a')as valf:
#                 valf.write(line)
# print(len(train))