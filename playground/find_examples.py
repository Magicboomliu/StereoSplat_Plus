import sys

def main(a_path, b_path):
    # 读 B.txt，放到一个 set 里方便查重
    with open(b_path, 'r') as f:
        b_names = {line.strip() for line in f if line.strip()}

    a = []
    # 逐行读 A.txt，如果不在 B 里就输出
    with open(a_path, 'r') as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            if name not in b_names:
                a.append(name)
    return a 

if __name__ == "__main__":
    filename1 = "/home/zliu/Project2025/FeedforwardGS_Ablations/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete.txt"
    filename2 = "/home/zliu/Project2025/FeedforwardGS_Ablations/FeedStereoGS/filenames/kitti360/trainval/val_2013_05_28_drive_0000_sync_complete_version2.txt"
    a = main(filename1, filename2)
    
    print(len(a))
    for name in a:
        print(name)
