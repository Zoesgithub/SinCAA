
python main.py \
 --experiment_name=n1_weight0.1_innl2 \
 --norm=GraphNorm \
 --num_head=16  \
 --learning_rate=1e-3 \
 --batch_size=640 \
 --model_channels=512 \
 --topological_net_layers=4 \
 --logger_step=1 \
 --train_aa_data_path=data/AAList/train_final_delta2.csv \
 --val_aa_data_path=data/AAList/val_final_delta2.csv \
 --num_epochs=20 \
 --num_workers=5  \
 --model=GPS \
 --max_combine=1 \
 --logger_step=10 \
 --decoder_layers=1 \
 --cont_weight=0.1 \
 --num_inner_l=2 
