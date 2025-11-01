cd tools/Complete-Striped-Smith-Waterman-Library/src
#python pyssw.py -p ../../../raw_data/fasta/peptides.fasta ../../../raw_data/fasta/peptides.fasta > ../../../raw_data/fasta/peptides_out2.txt 2>/dev/null
python pyssw.py -p ../../../raw_data/fasta/proteins.fasta ../../../raw_data/fasta/proteins.fasta > ../../../raw_data/fasta/proteins_out2.txt 2>/dev/null
cd ../../../
python tools/ss.py raw_data/fasta/peptides.fasta  raw_data/fasta/peptides_out2.txt peptides_mat.npy  peptides
python tools/ss.py raw_data/fasta/proteins.fasta raw_data/fasta/proteins_out2.txt proteins_mat.npy proteins

mv proteins_dict.dict raw_data/
mv proteins_mat.npy raw_data/
mv peptides_mat.npy raw_data/
mv peptides_dict.dict raw_data/