#this script combines individual dimension HF training .json files into a final combined HF training .json file

import sys
import json
import random

random.seed(42)


def combine_HF_files(prevention_file, treatment_file, risk_factor_file, output_file):
    prevention_f = open(prevention_file,'r',encoding='UTF-8')
    prevention_json = json.loads(prevention_f.read())
    print(len(prevention_json))
    
    treatment_f = open(treatment_file,'r',encoding='UTF-8')
    treatment_json = json.loads(treatment_f.read())
    print(len(treatment_json))
    
    risk_factor_f = open(risk_factor_file,'r',encoding='UTF-8')
    risk_factor_json = json.loads(risk_factor_f.read())
    print(len(risk_factor_json))
    
    output_lst = prevention_json + treatment_json + risk_factor_json
    print(len(output_lst))
    print(output_lst[:2])
    
    random.seed(42)
    random.shuffle(output_lst)
    print(len(output_lst))
    print(output_lst[:2])
    
    with open(output_file,'w',encoding='UTF-8') as out_f:
        json.dump(output_lst,out_f,indent=4)
    out_f.close()
    print("combined lines written to json file")


#prevention_file = 'prevention/prevention_test_HF.json'
#treatment_file = 'treatment/treatment_test_HF.json'
#risk_factor_file = 'risk-factor/risk-factor_test_HF.json'
#output_file = 'combined/combined_test_HF.json'

prevention_file = 'prevention/prevention_test-seen_HF.json'
treatment_file = 'treatment/treatment_test-seen_HF.json'
risk_factor_file = 'risk-factor/risk-factor_test-seen_HF.json'
output_file = 'combined/combined_test-seen_HF.json'

combine_HF_files(prevention_file, treatment_file, risk_factor_file, output_file)