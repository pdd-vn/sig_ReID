original repo: https://github.com/michuanhaohao/reid-strong-baseline.git  

## Prepare dataset
* Prepare your dataset at "data/dataset/":  
* The data structure should look like:  
```
data
  dataset  
      sig  
        id_1
          sig_1
          sig_2
          sig_3
          .
          .  
        id_2
          sig_1
          sig_2
          sig_3
          .
          .  
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
