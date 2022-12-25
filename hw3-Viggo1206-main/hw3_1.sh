if [ ! -f "./25_0.95.pth" ]; then
    wget https://www.dropbox.com/sh/pgcxclo4zr2s819/AADGSn2GiUWlBruSzouEr0ZPa?dl=0 -O checkpoint_p1.zip
    unzip ./checkpoint_p1.zip
fi
python3 hw3_1_testing.py -inputpath $1 -output $2