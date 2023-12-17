

file_path_1 = "/data/home/wangyouze/projects/others/MMT/Iterative-attack/outputs/results_MMT_1_2.txt"
file_path_2 = "/data/home/wangyouze/projects/others/MMT/Iterative-attack/outputs/results_MMT_2.txt"
file_path_3 = "/data/home/wangyouze/projects/others/MMT/Iterative-attack/outputs/results_MMT_3.txt"
file_path_4 = "/data/home/wangyouze/projects/others/MMT/Iterative-attack/outputs/results_MMT_4.txt"

f_res = open("/data/home/wangyouze/projects/others/MMT/Iterative-attack/outputs/results_MMT_all.txt", 'w', encoding='utf-8')
res = []
with open(file_path_1, 'r', encoding="utf-8") as pf:
        for i, line in enumerate(pf):
                res.append(line)

with open(file_path_2, 'r', encoding="utf-8") as pf:
        for i, line in enumerate(pf):
                res.append(line)

with open(file_path_3, 'r', encoding="utf-8") as pf:
        for i, line in enumerate(pf):
                res.append(line)
with open(file_path_4, 'r', encoding="utf-8") as pf:
        for i, line in enumerate(pf):
                res.append(line)
f_res.write(''.join(res))