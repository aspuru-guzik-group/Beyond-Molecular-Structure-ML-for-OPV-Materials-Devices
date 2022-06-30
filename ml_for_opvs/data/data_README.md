# README for data processing
## How to generate all input representations with new data
### If you modify the OPV data (i.e. add new data points, or new features, you'll want to remake the files with different input representations)
Run the following code in the correct directory.
<br>`python auto_generate_data.py`</br>


## If new donors or acceptors are used, please go to ml_for_opvs>data>input_presentation>manual_frag>manual_frag.py
Go to the bottom of the file and uncomment
`fragment_files("donor")`. This will allow you to manually fragment new donors. <br>Follow the prompts and look at the images produced. The index on each atom should guide you to fragment the correct bonds. (Familiarity with RDKiT is a huge plus) </br>
