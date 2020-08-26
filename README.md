original repo: https://github.com/michuanhaohao/reid-strong-baseline.git  
pretrained model: https://drive.google.com/drive/folders/12kdsYyit1hUn1BvrI87Swn8DAXBC9ZtP?usp=sharing

## Prepare dataset
* Prepare your dataset at 'dataset/'.  
* The data structure would like:  
```
dataset  
    sig  
      id1  
      id2  
      ...
```
## Train
* Reconfig `train.sh` if needed, see 'config/' for default configs and 'configs/' for some particular configs.  
* Remember to set `OUTPUT_DIR` in `train.sh` to your output directory.   
* Run `sh train.sh`.  

## Test
* Open `test.py`, pass your model path to `Sig_Ver_Model` when create a new one.  
* Using `veirfy` to verify 2 signature images using path, or `verify2` for 2 PIL image.  
* Using `multiple_pair_verify` to verify pairwise signature images of 2 folder.  
