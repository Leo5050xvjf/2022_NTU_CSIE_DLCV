if [ ! -f "./checkpoint.zip" ]; then
    wget https://www.dropbox.com/sh/hc9ue5o3b76kw6x/AADI3bseAhBXM-u0YU_1lf_ha?dl=0 -O checkpoint.zip
    unzip ./checkpoint.zip -x /
fi

python3 hw4_1_test.py --test_csv $1 --test_data_dir $2 --testcase_csv $3 --output_csv $4