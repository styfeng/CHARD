import os
import sys

#gen_folder = sys.argv[1] #gen_folder = "commongen_data/new_K2T_BART_MI/final_outputs_single/generations"

#gen_file_lst = os.listdir(gen_folder)
#gen_file_lst = sorted(gen_file_lst)
#print(len(gen_file_lst))
#print(gen_file_lst)

#for fn in gen_file_lst:
#print("\ncurrently evaluating {}".format(fn))
#os.system('python eval.py --key_file data/commongen_test_split_src_NEW.txt --gts_file data/commongen_test_split_tgt_NEW.txt --res_file {}'.format(input_path))
input_path = 'Accenture_data/T5-large-separate-dimensions/test-combined_generated_T5-large_1e-05_42_combined_2x_temp0.6_cp738_txt.txt_prevention.txt'
os.system('python eval_individual_BLEU.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --res_file {}'.format(input_path))
os.system('python eval_individual_meteor.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --res_file {}'.format(input_path))
os.system('python eval_individual_cider.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --res_file {}'.format(input_path))
os.system('python eval_individual_spice.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --res_file {}'.format(input_path))

input_path = 'Accenture_data/T5-large-separate-dimensions/test-combined_generated_T5-large_1e-05_42_combined_2x_temp0.6_cp738_txt.txt_treatment.txt'
os.system('python eval_individual_BLEU.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_treatment.txt --res_file {}'.format(input_path))
os.system('python eval_individual_meteor.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_treatment.txt --res_file {}'.format(input_path))
os.system('python eval_individual_cider.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_treatment.txt --res_file {}'.format(input_path))
os.system('python eval_individual_spice.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_treatment.txt --res_file {}'.format(input_path))

input_path = 'Accenture_data/T5-large-separate-dimensions/test-combined_generated_T5-large_1e-05_42_combined_2x_temp0.6_cp738_txt.txt_risk-factor.txt'
os.system('python eval_individual_BLEU.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_risk-factor.txt --res_file {}'.format(input_path))
os.system('python eval_individual_meteor.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_risk-factor.txt --res_file {}'.format(input_path))
os.system('python eval_individual_cider.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_risk-factor.txt --res_file {}'.format(input_path))
os.system('python eval_individual_spice.py --key_file Accenture_data/combined_test_HF_outputs.txt_prevention.txt --gts_file Accenture_data/combined_test_HF_outputs.txt_risk-factor.txt --res_file {}'.format(input_path))