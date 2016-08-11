#!/bin/bash
mkdir ./scripts

for seed in  {383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541}
do 
	cp train_hot_split2class_texture.lua  ./scripts/train_hot_split2class_texture_$seed".lua"
	echo "perl -wi -p -e 's/\(541\)/\("$seed"\)/g' ./scripts/train_hot_split2class_texture_"$seed".lua"
	echo "perl -wi -p -e 's/CSV_541/CSV_"$seed"/g' ./scripts/train_hot_split2class_texture_"$seed".lua"
done

