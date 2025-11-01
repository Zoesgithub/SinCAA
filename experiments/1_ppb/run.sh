for clu in 0.3 0.2 0.1 0.4
do
for exs in novel_protein novel_peptide both_9
do
ls data/peptide_data/data_source_${clu}/${exs}| while read line; do
python main.py --exper_setting ${exs} --fold ${line} --dist_threshold=1.543 --num_workers 1 --batch_size 1 --learning_rate=1e-4 --processed_dir=data/peptide_data/data_source_${clu}/
done
done
done