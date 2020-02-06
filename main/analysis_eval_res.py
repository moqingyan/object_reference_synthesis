import json 
import os 

def gen_report(data_dir):
    report = {}
    for cont_res_filename in os.listdir(data_dir):
        print(cont_res_filename)
        
        if not "iter_200" in cont_res_filename:
            continue 
        file_num = os.listdir(os.path.join(data_dir, cont_res_filename))
        if len(file_num) == 0:
            continue

        report[cont_res_filename] = {}

        old_version = set()
        new_version = set()
        latest_version = set()
        processed = []
        suc_method = {}

        cont_res_dir = os.path.join(data_dir, cont_res_filename)
        for prog_filename in os.listdir(cont_res_dir):
            prog_path = os.path.join(cont_res_dir, prog_filename)
            with open(prog_path, 'r') as prog_file:
                info = prog_file.read()
                if "model" in info:
                    suc_method[prog_filename] = "model"
                elif "exploit: 1" in info:
                    suc_method[prog_filename] = "d1"
                elif "exploit: 2" in info:
                    suc_method[prog_filename] = "d2"
                elif "exploit" in info:
                    suc_method[prog_filename] = "d1"
                elif "[]" in info:
                    suc_method[prog_filename] = "fail"
                else:
                    suc_method[prog_filename] = "unknown"
            
            if not "_1" in prog_filename:
                old_version.add(prog_filename)
            else:
                new_version.add(prog_filename)
                processed.append(prog_filename)

        overlaps = []

        for ct in processed:
            if (str(ct) + "_1") in new_version:
                latest_version.add(str(ct) + "_1")
                if (str(ct) in old_version) and not(suc_method[str(ct)] == "fail"):
                    overlaps.append(suc_method[str(ct) + "_1"])
            else:
                latest_version.add(str(ct))

        print(overlaps)
        
        model = 0
        unknown = 0
        d1 = 0
        d2 = 0
        fail = 0
        for prog in latest_version:
            sm = suc_method[prog]
            if sm == "model":
                model += 1
            elif sm == "d1":
                d1 += 1
            elif sm == "d2":
                d2 += 1
            elif sm == "fail":
                fail += 1
            else:
                unknown += 1 

        total = model + d1 + d2 + fail + unknown
        report[cont_res_filename]["model"] = model / total
        report[cont_res_filename]["d1"] = d1 / total
        report[cont_res_filename]["d2"] = d2 / total
        report[cont_res_filename]["fail"] = fail / total
        report[cont_res_filename]["unknown"] = unknown / total
        report[cont_res_filename]["process"] = len(processed) / 3500
        
    print (report)
    return report

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/eval_result/empty"))
    gen_report(data_dir)