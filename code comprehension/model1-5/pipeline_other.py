from tqdm import *
import subprocess
#auglist=[str(i) for i in range(1,10)]
#for aug in tqdm(auglist):
subprocess.run(['python','stim_evaluation.py','--problem-setting=subjective_difficulty','--split=code-snippet','--mode=code','--simulation=code','--seed=43','--output=../final_result_other.csv'])
subprocess.run(['python','stim_evaluation.py','--problem-setting=subjective_difficulty','--split=code-snippet','--mode=bimodal','--simulation=randomall','--seed=43','--output=../final_result_other.csv'])
subprocess.run(['python','stim_evaluation.py','--problem-setting=subjective_difficulty','--split=code-snippet','--mode=bimodal','--simulation=real','--seed=43','--output=../final_result_other.csv'])
subprocess.run(['python','stim_evaluation.py','--problem-setting=subjective_difficulty','--split=code-snippet','--mode=bimodal','--simulation=0','--seed=43','--output=../final_result_other.csv'])


