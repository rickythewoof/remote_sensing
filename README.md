# Repository for remote sensing work

### Workflow
Using python
install dependencies with 
```sudo apt install awscli && aws configure ```
```pip  install numpy pandas matplotlib scipy sklearn ```



### Get Dataset

_it's BIG, don't use mobile data_

    aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz .

### Sources
[Spacenet AI](https://spacenet.ai/sn6-challenge/)