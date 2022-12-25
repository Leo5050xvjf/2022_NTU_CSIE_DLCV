wget 'https://www.dropbox.com/s/bb0ide7ww5viqn2/checkpoint_p1.zip?dl=0'  -O checkpoint_p1.zip

# Unzip the downloaded zip file
unzip ./checkpoint_p1.zip

# Remove the downloaded zip file
rm ./checkpoint_p1.zip

python3 hw1_1_testing.py -t $1 -o $2
