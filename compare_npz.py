import os
import sys
import numpy as np

from sg_custom_modules.fast_core import run_fast_infer

def main():
    samples = ["sample1", "sample2", "sample3", "sample4"]
    for sample in samples:
        image_path = f"base_meta_code/assets/qualitative_comparisons/{sample}/input_bbox.png"
        ckpt_path = "checkpoints/sam-3d-body-dinov3/model.ckpt"
        mhr_path = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
        
        out_dir_full = f"output/compare/{sample}/full"
        out_dir_body = f"output/compare/{sample}/body"
        
        os.makedirs(out_dir_full, exist_ok=True)
        os.makedirs(out_dir_body, exist_ok=True)
        
        print(f"\n================ Processing {sample} ================")
        print("Running inference with type 'full'...")
        summaries_full = run_fast_infer(
            input_path=image_path,
            output_dir=out_dir_full,
            checkpoint_path=ckpt_path,
            mhr_path=mhr_path,
            inference_type="full"
        )
        res_full = summaries_full[0]["out_npz"]
        
        print("Running inference with type 'body'...")
        summaries_body = run_fast_infer(
            input_path=image_path,
            output_dir=out_dir_body,
            checkpoint_path=ckpt_path,
            mhr_path=mhr_path,
            inference_type="body"
        )
        res_body = summaries_body[0]["out_npz"]
        
        print(f"Comparing {res_full} and {res_body}...")
        npz_full = dict(np.load(res_full, allow_pickle=False))
        npz_body = dict(np.load(res_body, allow_pickle=False))
        
        print(f"Keys in full:  {len(npz_full.keys())}")
        print(f"Keys in body:  {len(npz_body.keys())}")
        
        differences = {}
        for k in npz_full.keys():
            if k not in npz_body:
                differences[k] = "Missing in body"
                continue
                
            val_full = npz_full[k]
            val_body = npz_body[k]
            
            if val_full.shape != val_body.shape:
                differences[k] = f"Shape mismatch: {val_full.shape} vs {val_body.shape}"
                continue
                
            if issubclass(val_full.dtype.type, np.inexact):
                # Float comparison
                max_diff = np.max(np.abs(val_full - val_body))
                if max_diff > 1e-5:
                    differences[k] = f"Values differ, max absolute difference: {max_diff}"
            else:
                if not np.array_equal(val_full, val_body):
                    differences[k] = "Categorical/Integer arrays differ"
                    
        for k in npz_body.keys():
            if k not in npz_full:
                differences[k] = "Missing in full"
                
        if differences:
            print("Differences found:")
            for k, diff in differences.items():
                print(f"  {k}: {diff}")
        else:
            print("Both outputs are IDENTICAL.")

if __name__ == "__main__":
    main()
