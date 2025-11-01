xargs -P 20 -I {} -d '\n' python -m tools.prepare_data --line {} < raw_data/make_data/all_data.txt
