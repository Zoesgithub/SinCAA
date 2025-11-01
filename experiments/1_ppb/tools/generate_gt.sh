#xargs -P 20 -I {} -d '\n' python -m tools.prepare_data --cutoff 0.657 0.956 1.543 1.983 2.443 --line {} < raw_data/make_data/all_data.txt
xargs -P 20 -I {} -d '\n' python -m tools.prepare_data --cutoff 2 --line {} < raw_data/make_data/all_data.txt
