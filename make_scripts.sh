#!/bin/bash
mkdir ./scripts

for seed in  {383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541}
do 
#copy the script files into various copies each for different seed
	cp train_hot_split2class_texture.lua  ./runs/train_hot_split2class_texture_$seed".lua"
	cp ./runs/mds.R ./runs/mds_$seed".R"
	cp ./runs/plotpoint.R ./runs/plotpoint_$seed".R"

	echo "perl -wi -p -e 's/\(541\)/\("$seed"\)/g' ./runs/train_hot_split2class_texture_"$seed".lua"
	echo "perl -wi -p -e 's/CSV_541/CSV_"$seed"/g' ./runs/train_hot_split2class_texture_"$seed".lua"
	echo "perl -wi -p -e 's/CSV_541/CSV_"$seed"/g' ./runs/mds_"$seed".R"
	echo "perl -wi -p -e 's/CSV_541/CSV_"$seed"/g' ./runs/plotpoint_"$seed".R"
	#above commands generate the perl commands to replace the text in the R and sh script.
	#replaced text is the seed value and folder names. s - substitute. g - all instances. 
done

