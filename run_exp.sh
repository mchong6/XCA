#!/bin/bash
mkdir ./runs
cd ./runs
	#remove the symbolic links in the runs folder  if present
	rm ./provider_texture.t7 ./provider_texture.lua

	#creat the symbolic links in the runs folder to the provider files in the parent directory of runs folder
	ln -s ../provider_texture.t7 ./provider_texture.t7
	ln -s ../provider_texture.lua ./provider_texture.lua
for seed in  {383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541}
do 
	echo $seed
	pwd
	mkdir ./CSV_$seed
	#create the output folder
	#cp  ../scripts/train_hot_split2class_texture_$seed".lua" .
	#cp  ../scripts/mds_$seed".R" .

	#copy the sh and R scripts
	th ./train_hot_split2class_texture_$seed".lua"
	echo "th done for seed:"$seed
	Rscript  ./mds_$seed".R"
	echo "Rscript done for seed:"$seed
	Rscript  ./plotpoint_$seed".R"
	echo "Rscript done for plotpoint:"$seed
done
cd ..

