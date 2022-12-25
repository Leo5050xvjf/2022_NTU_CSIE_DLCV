wget https://www.dropbox.com/sh/st67wclcf621kpi/AAApc360SqauwB_QK1w8rAIsa?dl=0 -O checkpoint_p2.zip
unzip ./checkpoint_p2.zip -x /
python3 predict.py -inputpath $1 -outputpath $2