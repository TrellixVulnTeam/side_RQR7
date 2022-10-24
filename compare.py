import os
import sys
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--case", default = "res50_tf_npu", type = str)
args = parser.parse_args()

path = "/workspace/vsi_training_compare"


cpu_path = path + "/resnet_cpu"
npu_path = path + "/" + args.case
diff_path = path + "/" + args.case + "_diff"
if not os.path.exists(diff_path):
    os.mkdir(diff_path)

cpu_before_path = cpu_path + "/before"
npu_before_path = npu_path + "/before"
# diff_before_path = diff_path + "/before"
# if not os.path.exists(diff_before_path):
#     os.mkdir(diff_before_path)

cpu_after_path = cpu_path + "/after"
npu_after_path = npu_path + "/after"
diff_after_path = diff_path + "/after"
if not os.path.exists(diff_after_path):
    os.mkdir(diff_after_path)

cpu_grads_path = cpu_path + "/grads"
npu_grads_path = npu_path + "/grads"
# diff_grads_path = diff_path + "/grads"
# if not os.path.exists(diff_grads_path):
#     os.mkdir(diff_grads_path)

before_file_list = os.listdir(cpu_before_path)
after_file_list = os.listdir(cpu_after_path)
after_file_list.remove("resnet.h5")
grads_file_list = os.listdir(cpu_after_path)
###test for before

print("#############test for before data############")
for before_file in before_file_list:
    cpu_before_file_full = cpu_before_path + "/" + before_file
    npu_before_file_full = npu_before_path + "/" + before_file

    with open(cpu_before_file_full, 'r') as cpu_before_file_context:
        cpu_before_file_lists = cpu_before_file_context.readlines()

    with open(npu_before_file_full, 'r') as npu_before_file_context:
        npu_before_file_lists = npu_before_file_context.readlines()

    size = len(cpu_before_file_lists)
    for i in range(size):
        if cpu_before_file_lists[i] != npu_before_file_lists[i]:
            print("npu cpu data is not same")
            sys.exit(0)
print("before data is right")

###test for after
print("#############test for after data############")
ok_case = []
fail_case = []
for after_file in after_file_list:
    # print(after_file)
    cpu_after_file_full = cpu_after_path + "/" + after_file
    npu_after_file_full = npu_after_path + "/" + after_file
    diff_after_file_full = diff_after_path + "/" + after_file
    f=open(diff_after_file_full,'w')
    with open(cpu_after_file_full, 'r') as cpu_after_file_context:
        cpu_after_file_lists = cpu_after_file_context.readlines()

    with open(npu_after_file_full, 'r') as npu_after_file_context:
        npu_after_file_lists = npu_after_file_context.readlines()

    size = len(cpu_after_file_lists)
    for i in range(size):
        diff_num = float(cpu_after_file_lists[i]) - float(npu_after_file_lists[i])
        f.write(str(diff_num))
        f.write('\n')
    f.close()
    n1 = np.loadtxt(diff_after_file_full, dtype=np.float64)
    std = n1.std()
    if std < 0.00001:
        ok_case.append(after_file)
    else:
        fail_case.append((after_file, std))
    # print("%s Diff: Max: %f - %d, Min: %f - %d, STD: %f" % 
    #     (after_file, n1.max(), n1.argmax(), n1.min(), n1.argmin(), std))
print("==================== OK Ratio {:.2f}% ===================".format(100*(len(ok_case)/len(after_file_list))))
with open(diff_path + "/compare_result.txt", 'w') as fp:
    fp.write("OK Ratio {:.2f}\n".format(100*(len(ok_case)/len(after_file_list))))
    fp.write("\n".join(str(item) for item in ok_case))
    fp.write("\n=== Fail: === \n")
    for name, value in fail_case:
        fp.write("{}: std: {}\n".format(name, value))
print("RRR : job finish.")
"""
###test for grads
print("#############test for grads data############")
for grads_file in grads_file_list:
    print(grads_file)
    cpu_grads_file_full = cpu_grads_path + "/" + grads_file
    npu_grads_file_full = npu_grads_path + "/" + grads_file
    diff_grads_file_full = diff_grads_path + "/" + grads_file
    f=open(diff_grads_file_full,'w')

    with open(cpu_grads_file_full, 'r') as cpu_grads_file_context:
        cpu_grads_file_lists = cpu_grads_file_context.readlines()

    with open(npu_grads_file_full, 'r') as npu_grads_file_context:
        npu_grads_file_lists = npu_grads_file_context.readlines()

    size = len(cpu_grads_file_lists)
    for i in range(size):
        diff_num = float(cpu_grads_file_lists[i]) - float(npu_grads_file_lists[i])
        f.write(str(diff_num))
        f.write('\n')
        # if cpu_grads_file_lists[i] != npu_grads_file_lists[i]:
        #     print(cpu_grads_file_lists[i],end='')
        #     print(npu_grads_file_lists[i])
"""