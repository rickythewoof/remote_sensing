# Repository for remote sensing work
Repository made for Artificial Intelligence & Neural Network course with Prof. Ciarfuglia @ Sapienza 

### Install required packages
Install everything with 

```console
sudo apt install awscli python3 python3-pip && aws configure 
pip install -r src/requirements.txt
```

### Get Dataset
_it's BIG, don't use mobile data_
```console
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz data 
gunzip SN6_buildings_AOI_11_Rotterdam_train.tar.gz
```

### Sources
[Spacenet AI](https://spacenet.ai/sn6-challenge/)
[Neural Network lessons](https://youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)