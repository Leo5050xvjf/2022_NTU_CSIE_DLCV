if [ ! -f "./checkpoint.zip" ]; then
    wget https://www.dropbox.com/sh/hc9ue5o3b76kw6x/AADI3bseAhBXM-u0YU_1lf_ha?dl=0 -O checkpoint.zip
    unzip ./checkpoint.zip -x /
fi
python3 hw4_p2_testing.py --testCSV $1 --testImage $2 --output $3