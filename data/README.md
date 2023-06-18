Relevant data for the paper: "CHARD: Clinical Health-Aware Reasoning Across Dimensions for Text Generation Models" (https://arxiv.org/abs/2210.04191)
Work by Carnegie Mellon University and Accenture Labs


CHARDAT folder contains data for our actual CHARDAT dataset, separated into three subfolders for each of the three dimensions.

Each of those subfolders contains three .txt files for the train, val, and test splits. Each line contains an example in the format "input_ID <sep> condition <sep> dimension attribute <sep> passage containing explanation". 

The input_ID is the ID of that specific example when collecting the data from AMT, and is a unique identifier for each example, but can be ignored for most purposes.



HF_training_data folder contains the .json files used to train BART and T5 and run inference + evaluation using BART and T5 (using their HuggingFace codebases). It is split into the three individual dimensions (for training dimension-specific models), a "combined" subfolder for training the final combined models, and an "augmented" folder for training the data augmented combined models.

In the augmented subfolder, the "2x" subfolder contains 2x augmentation training data for various backtranslation temperatures. The other subfolders contain training data for the different data augmentation strategies and amounts (see the paper for more information).



input_GT-output_txt_files folder contains .txt files with the inputs and ground-truth outputs (one per line) for the various test splits (combined, for each of the three dimensions, and further split into the test-seen and test-unseen portions of each test split). These .txt files allow for easy viewing of the inputs and ground-truth outputs (one per line) for each test split, and are used for evaluation purposes (e.g. to get BLEU and other metrics).



all_generation_txt_files folder contains the final outputs by the various models/methods, split by various attributes (e.g. combined vs. individual dimensions, the different methods/models, etc.). 

"humans" subfolder contains the ground-truth human-written outputs, "retrieval" subfolder contains the Google Search retrieval-based baseline outputs, and "best_T5_model" subfolder contains the outputs by the best T5-large model (which was subsequently used for qualitative analysis and human evaluation). 

The remaining folders ("prevention, "risk-factor", "treatment", and "combined") contain various model outputs split by the test-split (either full test-split (named "test-combined"), test-seen, or test-unseen).

The "combined" subfolder contains outputs by several different models on the combination of all three dimensions. See the paper for more info, and to determine the "best" models per model type (BART vs. T5) and size (base vs. large) combination.



backtranslation_data folder contains all the data used for backtranslation. The "inputs" subfolder contains the input .txt files to backtranslation. Specifically, "combined_train_explanations.txt" contains the explanation portion of examples that is actually backtranslated, whereas "combined_train_initials.txt" contains the initial part of the passages/templates which does not get backtranslated.

The "backtranslations" subfolder contains actual UDA backtranslated explanations for various backtranslation temperatures, including different versions (e.g. up to 9 for temp=0.9). See the paper for more info.