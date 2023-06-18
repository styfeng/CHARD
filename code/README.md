See "requirements_and_other_info.txt" for important info about required packages/libraries, python versions, conda environments, and so forth for each part of this project.



BART-T5-scripts folder contains scripts related to training BART and T5 and running inference + evaluation on them.


The "training" subfolder contains scripts for training BART and T5. More detailed instructions (and explanations of each argument) can be found as comments in the two python files. Examples commands are as follows:

python bart_training.py base 3e-05 42 32 128 combined_baseline HF_training_data/combined/combined_train_HF.json HF_training_data/combined/combined_val_HF.json 32 30 0 

python t5_training.py base 1e-05 42 35 128 combined_2x_temp0.4 HF_training_data/augmented/2x/combined_train_temp0.4_2x.json HF_training_data/combined/combined_val_HF.json 64 25 0

loop_training.sh is a script that allows you to train over several different learning rates with a single command. You give a command such as the one below:

bash loop_training.sh t5 large 42 35 128 combined_diff-temps_3x HF_training_data/augmented/diff-temps-t5/combined_train_diff-temps-t5_3x.json HF_training_data/combined/combined_val_HF.json 16 12 0

Then you will be prompted to give a list of learning rates separated by a 'space' between each, such as: 1e-05 5e-05 1e-04 5e-04 1e-06 5e-06 1e-03 5e-03 5e-07 1e-02

The script then will train a version of the given model (defined by arguments to the script) for each of the given learning rates. See the script for more information on the arguments.


The "inference_evaluation" subfolder contains several scripts for running inference and subsequently evaluation on the trained models.

"bart_inference_evaluate.py" and "t5_inference_evaluate.py" load a given model, generate outputs using that given model with beam search decoding, and then calculate a subset of evaluation metrics: ROUGE, BERTScore, PPL, Len, TTR, UTR. The outputs are saved to two files: one .json (containing inputs, expected ground-truth outputs, and the model's outputs) and one .txt file (that simply contains the model's generations, one per line). Files with the individual metrics per example are also saved, which can be useful for statistical significance calculations and other purposes later. Lastly, a file with the overall average statistics is outputted. See comments in the files themselves for more info, including info about each argument.

A sample command for the above scripts is as follows:

python bart_inference_evaluate.py base 42 32 128 risk-factor_baseline HF_training_data/risk-factor/risk-factor_test_HF.json 3e-03 40 32 0


The "inference_three_test_splits.sh" files (4 total, one for combined, and three more for the three individual dimensions) essentially runs either "bart_inference_evaluate.py" or "t5_inference_evaluate.py" but on three separate test splits: the full combined test split, the test-seen half, and the test-unseen half. It essentially streamlines the process by running inference and evaluation on all three test split variations at once. See the script for more information on the arguments.

A sample command is as follows:

bash BART-T5-scripts/inference_three_test_splits_prevention.sh t5 base 42 35 128 prevention_baseline 3e-03 12 32 0


"evaluate_from_file.py" evaluates the aforementioned metrics given an input .json file that already contains the model's generations, rather than loading a model, generating from the model, then evaluating. "eval_diversity_PPL_from_json_file.py" and "eval_diversity_PPL_from_txt_file.py" are similar, and are specifically for evaluating diversity and PPL metrics from a file (either .json or .txt file, respectively) with the generations. These are specifically used when there are no ground-truth outputs to compare to, and hence ROUGE and BERTScore metrics are avoided - e.g. if you want to calculate the diversity and PPL of the human-written ground-truth outputs themselves. These files all take a single argument which is the name or path of the file that contains the generations to be evaluated.

"ROUGE_BERTScore_from_files.py" calculates the ROUGE and BERTScore between two sets of outputs contained in two separate files, which are the two arguments.



commongen_eval folder contains scripts to evaluate the remaining metrics: BLEU, METEOR, CIDEr, and SPICE. It is named "commongen_eval" because we use the CommonGen authors' evaluation scripts found here: https://github.com/INK-USC/CommonGen. Please clone this repo, then navigate to "evaluation/Traditional/eval_metrics". 

In this folder, you can paste in all the scripts contained in "commongen_eval", and make a subfolder called "Accenture_data" where you can paste in .txt files with the various inputs to the models (e.g. "combined_test_HF_inputs.txt") and expected ground-truth outputs (e.g. "combined_test_HF_outputs.txt"). These can all be found in "data/input_GT-output_txt_files". You can also create new subfolders in which you can move several .txt files containing model outputs or generations.

The "eval_individual_X.py" files (where X is a metric like BLEU) evaluates the metric X and also saves the metric results per individual example/line to a file (e.g. for statistical significance evaluation purposes later). The subfolder "loop_scripts" contains scripts that loops through a folder that contains several output .txt files (this folder is the input argument to each script), and runs all the "eval_individual_X.py" files on all the output .txt files in this folder. Look into each script for more information on the files/arguments it expects and how exactly it works.



data_processing folder contains scripts to process data for the project: either preprocessing, postprocessing, or other processing. It is split into subfolders which are named depending on the particular application/purpose, and the names should be self-explanatory. Each file has a brief comment at the top which explains its general purpose and usage, so please look at the individual scripts for more information.



UDA_backtrans folder contains code for UDA backtranslation (for data augmentation), based off the UDA repo: https://github.com/google-research/uda. Some key files can be found in this folder, but the majority of files are found in that repo and it should be cloned. 

"main_notebook_UDA_backtranslation.ipynb" is a jupyter notebook (based off a colab notebook) that contains all the commands and instructions to run UDA backtranslation. Please look at the notebook and the instructions for all the required information (it should take care of everything).