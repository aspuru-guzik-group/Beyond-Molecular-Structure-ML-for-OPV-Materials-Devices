# README for data processing
## How to generate all input representations with new data
### If you modify the OPV data (i.e. add new data points, or new features, you'll want to remake the files with different input representations)
0. Download Google Sheets as .csv into ml_for_opvs/ml_for_opvs/data/raw/OPV_Min/Machine Learning OPV Parameters - device_params.csv
1. Run the following code from the correct directory containing this python file.
<br>`python auto_generate_data.py`</br>
All input representations will be created.
2. Filter out datapoints by the appropriate features (ex. device, electronic_only, fabrication, etc.) using \*_feat_select.py from ml_for_opvs/ml_for_opvs/data/input_representation/\*_feat_select.py (\* indicates any input_representation)
3. Generate KFold cross-validation from ml_for_opvs/ml_for_opvs/data/input_representation/cross_validation.py 
<br> Go to directory with desired data file. (Example: ml_for_opvs/ml_for_opvs/data/input_representation/BRICS/master_brics_frag.csv)
Example ran in terminal: </br> `python ../../cross_validation.py --dataset_path ~/ml_for_opvs/ml_for_opvs/data/input_representation/OPV_Min/BRICS/master_brics_frag.csv --num_of_folds 5 --type_of_crossval KFold`
4. Good to go for training models!


## If new donors or acceptors are used, please go to ml_for_opvs/ml_for_opvs/data/input_presentation/manual_frag/manual_frag.py
Go to the bottom of the file and uncomment
`fragment_files("donor")`. This will allow you to manually fragment new donors. <br>Follow the prompts and look at the images produced. The index on each atom should guide you to fragment the correct bonds. (Familiarity with RDKiT is a huge plus) </br>
