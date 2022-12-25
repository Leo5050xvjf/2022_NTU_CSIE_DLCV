wget "https://www.dropbox.com/s/oad6ybtb5yv1ay1/checkpoint_p2.zip?dl=0" -O checkpoint_p2.zip

# Unzip the downloaded zip file
unzip ./checkpoint_p2.zip

# Remove the downloaded zip file
rm ./checkpoint_p2.zip

python3 hw1_2_testing.py -t $1 -o $2
