# Download the main datasets.
curl https://raw.githubusercontent.com/AliOsm/arabic-text-diacritization/master/dataset/train.txt -o data/train.txt
curl https://raw.githubusercontent.com/AliOsm/arabic-text-diacritization/master/dataset/val.txt -o data/val.txt
curl https://raw.githubusercontent.com/AliOsm/arabic-text-diacritization/master/dataset/test.txt -o data/test.txt

# Download and extract the extra train dataset.
curl -L https://raw.githubusercontent.com/AliOsm/shakkelha/master/dataset/extra_train.zip -o data/extra_train.zip
unzip data/extra_train.zip -d data

# Merge the main train dataset with the extra train dataset.
cat data/train.txt data/extra_train.txt > data/merged_train.txt
mv data/merged_train.txt data/train.txt

# Clean up the extra train dataset.
rm data/extra_train.zip data/extra_train.txt
