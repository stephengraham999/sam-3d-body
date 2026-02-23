# fast_core.py - Technical Documentation

------------------------------------------------------------
## SECTION 1 — What This Script Does (For End Users)
------------------------------------------------------------

- **What problem does this script solve?** 
  It takes 2D images (like photographs of people) and uses an advanced AI model (SAM 3D Body) to estimate their full 3D body pose, shape, and camera perspective. Crucially, it wraps this complex AI model in a high-performance, production-ready system that processes large batches of images extremely fast without crashing or wasting time re-processing images it has already seen.

- **What does it produce?**
  It produces mathematical 3D data files (`.npz` files) for each input image. These files contain the 3D coordinates (vertices and keypoints) that make up the human body mesh, along with camera parameters.

- **When would someone use it?**
  A user would use this when they have a folder containing hundreds or thousands of photos of people and they need to extract 3D human meshes from all of them as quickly as possible, perhaps to train another AI, power an animation system, or perform biomechanical analysis.

- **What type of script is it?**
  It is a high-performance **batch processing utility / API module**. It is designed to be imported and called by other scripts (like a command-line wrapper or a GUI application), rather than being run directly from the command line itself.

------------------------------------------------------------
## SECTION 2 — How To Use It (End Users)
------------------------------------------------------------

### 1. How to run it
This script is not meant to be run directly via the command line (it has no `if __name__ == "__main__":` block or `argparse` setup). Instead, it must be imported and called from another Python script (e.g., `demo.py` or your own custom script).

**Example Usage in Python:**
```python
from sg_custom_modules.fast_core import run_fast_infer

summaries = run_fast_infer(
    input_path="./my_images",
    output_dir="./my_outputs",
    checkpoint_path="./checkpoints/sam-3d-body-dinov3/model.ckpt",
    mhr_path="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
    inference_type="full",
    cache_dir="./my_cache" # Optional, but highly recommended
)
```

**Required arguments:**
- `input_path`: String or list of strings. Can be a single image file, a directory of images, or a list of specific image paths.
- `output_dir`: String. The folder where the `.npz` files will be saved.
- `checkpoint_path`: String. Path to the main model weights file (`.ckpt`).
- `mhr_path`: String. Path to the Momentum Human Rig model weights (`.pt`).

**Optional arguments:**
- `max_side`: (Default: `1280`) The maximum resolution to resize images to before processing.
- `single_person`: (Default: `True`) If multiple people are in the photo, only keep the largest one.
- `map_2d_to_original`: (Default: `True`) Scales output 2D coordinates back to match the original image size.
- `save_compressed`: (Default: `False`) Saves `.npz` files with gzip compression to save disk space (but is slower to save/load).
- `cache_dir`: (Default: `""`) Directory to store cached outputs. **Highly recommended** to speed up repeated runs.
- `bbox_thr`: (Default: `0.8`) Confidence threshold for the human detector.
- `inference_type`: (Default: `"full"`) Use `"full"` for high-quality body + hand reconstruction, or `"body"` for faster body-only reconstruction.
- `progress_fn`: (Default: `None`) A callback function to hook into for UI progress bars.

### 2. What inputs it expects
- **File types**: Images with extensions `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, or `.tiff`.
- **Directory structure**: The script will scan the `input_path` (if it's a directory) for the supported image extensions. It does not search recursively through subfolders.
- **Required dependencies**: `torch`, `numpy`, `cv2` (OpenCV), and the core `sam_3d_body` AI repository code must be accessible.
- **Environment setup**: Requires a system with a compatible compute device (NVIDIA CUDA GPU is highly recommended, Apple Silicon MPS is supported, CPU works but is very slow).

### 3. What outputs it creates
- **Files**: One `.npz` file per input image, saved inside `output_dir`. The filename will match the input image (e.g., `photo.jpg` -> `photo.npz`). If `cache_dir` is provided, cached copies will also be created there using long hexadecimal filenames.
- **Console output**: Prints device detection, progress exactly like `[1/100] photo.jpg | people=1 infer_ms=250.0...`, cache hits, and a final batch summary.
- **Side effects**: Creates output directories if they don't exist. Modifies the global `_ESTIMATOR_CACHE` to keep the loaded AI model in system memory.

### 4. Common mistakes users might make
- Expecting it to handle hundreds of people in a crowd photo well (the detector might struggle, and `single_person=True` throws out all but the largest).
- Running out of disk space by providing thousands of images and leaving `save_compressed=False`.
- Trying to run the script directly from the command line via `python fast_core.py`.
- Passing the wrong paths for the hefty model checkpoint files.

------------------------------------------------------------
## SECTION 3 — Technical Explanation (For Developers)
------------------------------------------------------------

- **Overall architecture of the file:**
  It is an asynchronous, cache-first batch inference pipeline. It strictly separates I/O (disk reads/writes) from GPU compute to maximize hardware utilization. It operates via three main stages: a quick cache-validation pass, a prefetch-and-inference loop using thread pools, and a dedicated background writer thread.

- **Main classes and functions:**
  - `run_fast_infer`: The master orchestrator function.
  - `_get_or_create_estimator`: A thread-safe singleton manager for keeping the heavy PyTorch model loaded in memory across multiple function calls.
  - `_prefetch_image` / `_save_npz_worker`: I/O background worker functions.
  - `to_npz_payload`: Standardizes the dictionary schema saved to disk.

- **Execution flow (what happens step-by-step):**
  1. Input paths are gathered and validated.
  2. If caching is enabled, an exact cache dictionary signature is built based on file contents and inference parameters.
  3. The script loops through all images. If a cache hit is found, it instantly copies the file to the output and skips AI processing for that image.
  4. Only if cache misses exist does the heavy AI model get loaded into memory.
  5. A background thread is started strictly for writing outputs to disk.
  6. A background thread pool begins reading and resizing the first image.
  7. The main loop begins: It claims the pre-fetched image, immediately schedules the prefetch for the *next* image, pushes the current image to the GPU for inference, filters the outputs, constructs the payload, and drops it into a queue for the background writer thread.
  8. Once all images are processed, poison pills are sent to the threads to gracefully shut them down.

- **Key algorithms or logic:**
  - **Caching logic:** Uses SHA-256 to hash the *content* of the input image (not the modification time, meaning it survives Git clones) along with all settings dicts. If the hash matches an existing file in `cache_dir`, processing is skipped completely.
  - **Asynchronous overlap:** By using `ThreadPoolExecutor` for reading/resizing and a `Queue` for saving, the GPU (which runs synchronously on the main thread) never sits idle waiting for a hard drive to spin or for JPEG decoding to finish.

- **External libraries used and why:**
  - `cv2` (OpenCV): De-facto standard for exceedingly fast C++ backed image reading and resizing.
  - `hashlib`: Generating secure SHA256 hashes for the caching engine.
  - `threading`, `queue`, `concurrent.futures`: Standard Python concurrency tools to build the async I/O pipeline.
  - `numpy`: Fast, compiled array processing and the required format (`.npz`) for saving tensor data.
  - `torch`: PyTorch, the core deep learning framework.

- **Any global state or threading:**
  - **Global State**: `_ESTIMATOR_CACHE` dictionary and `_ESTIMATOR_LOCK` thread lock. These safely keep the PyTorch model instantiated in global memory so consecutive calls to `run_fast_infer` in the same Python session don't have to wait 10 seconds for the model to load from disk again.
  - **Threading**: Extremely thread-heavy. The main function spawns a dedicated daemon `Thread` for saving files via a `queue.Queue`, and uses a `ThreadPoolExecutor` (max_workers=1) for prefetching images.

- **Error handling approach:**
  - It uses strict failure handling for missing inputs early in the process (`raise RuntimeError("No images found...")`).
  - It uses graceful failure for inner-loop issues. If an image is corrupt and OpenCV fails to read it (`img_bgr is None`), the prefetcher returns `None`, the main loop prints `[skip] unreadable`, and moves on without crashing the whole batch. Exceptions during the background save thread are caught, printed, and ignored.

------------------------------------------------------------
## SECTION 4 — Function & Parameter Breakdown
------------------------------------------------------------

### `detect_device()`
- **Purpose**: Automatically finds the best hardware accelerator available.
- **Parameters**: None.
- **Return value**: `torch.device`. Either "cuda", "mps", or "cpu".

### `list_images()`
- **Purpose**: Flattens a directory or single file path into a sorted list of supported image files.
- **Parameters**:
    - `input_path` (str, required): The target directory or file.
- **Return value**: `list[str]`. Sorted absolute or relative paths.

### `resize_keep_aspect()`
- **Purpose**: Downscales an image if its longest edge exceeds a threshold, calculating the scaling factors used to map mathematical coordinates back to the original size.
- **Parameters**:
    - `img_bgr` (np.ndarray, required): OpenCV BGR numpy array.
    - `max_side` (int, required): Maximum allowed pixel dimension.
- **Return value**: `tuple[np.ndarray, float, float]`. The resized image, the X scale factor, and Y scale factor.

### `pick_single_person()`
- **Purpose**: Filters a list of detected people down to just the single person taking up the most physical area in the bounding box.
- **Parameters**:
    - `outputs` (list[dict], required): The raw output list from the SAM 3D model.
- **Return value**: `list[dict]`. A list containing only 1 dictionary.

### `to_npz_payload()`
- **Purpose**: Converts the raw outputs into a strict dictionary schema suitable for `np.savez`.
- **Parameters**:
    - `outputs`, `faces`, `image_path`, `orig_w`, `orig_h`, `proc_w`, `proc_h`, `scale_x`, `scale_y`, `map_2d_to_original`, `infer_ms`.
    - (Self-explanatory formatting parameters passing through from the main loop).
- **Return value**: `dict[str, np.ndarray]`. A flat dictionary of numpy arrays.

### `_cache_key()`
- **Purpose**: Creates the unique identifier used for caching file names.
- **Parameters**:
    - `image_path` (str, required): Path to the image file to read and hash.
    - `context` (dict, required): JSON-serializable dictionary of inference settings.
- **Return value**: `str`. A 64-character SHA256 hex digest.
- **Side effects**: Heavy disk read (reads the entire image file sequentially to generate the hash).

### `_get_or_create_estimator()`
- **Purpose**: Thread-safe model loader/cache.
- **Parameters**:
    - `checkpoint_path` (str, required): Target weights file.
    - `mhr_path` (str, required): Target rig file.
- **Return value**: `tuple[SAM3DBodyEstimator, str]`. The initialized model object, and a string representing the device in use.
- **Side effects**: Modifies global memory state, heavily utilizes GPU VRAM upon load.

### `_prefetch_image()`
- **Purpose**: Async worker function to read and resize images off the main thread.
- **Parameters**:
    - `image_path` (str, required)
    - `max_side` (int, required)
- **Return value**: Optional 8-tuple containing the original OpenCV array, dimensions, the processed array, and scaling factors. Returns `None` if unreadable.
- **Side effects**: Disk read.

### `_save_npz_worker()`
- **Purpose**: Async daemon worker function to write dict arrays to disk.
- **Parameters**:
    - `save_queue` (queue.Queue, required): A thread-safe queue containing tuples of data to save.
- **Return value**: `None`
- **Side effects**: Constant background disk writes. Can duplicate files via `shutil.copy2` if caching is enabled.

### `run_fast_infer()`
- **Purpose**: The main orchestration pipeline.
- **Parameters**:
    - `input_path` (Union[str, list[str]], required): Image(s) to process.
    - `output_dir` (str, required): Where to save results.
    - `checkpoint_path` (str, required): AI model path.
    - `mhr_path` (str, required): AI rig path.
    - `max_side` (int, optional, 1280): Pre-processing downscale limit.
    - `single_person` (bool, optional, True): Filter multi-person outputs.
    - `map_2d_to_original` (bool, optional, True): Scale output matrices back to original resolution space.
    - `save_compressed` (bool, optional, False): Use `savez_compressed`.
    - `cache_dir` (str, optional, ""): Provide a path to enable hashing/caching.
    - `bbox_thr` (float, optional, 0.8): AI detection threshold.
    - `inference_type` (str, optional, "full"): Choose "full" or "body".
    - `progress_fn` (Callable, optional, None): UI hook.
- **Return value**: `list[dict]`. A list of metadata summaries for every file processed.
- **Side effects**: Spawns multiple threads. Heavy GPU usage. Massive disk I/O. Will overwrite pre-existing files in `output_dir`.

------------------------------------------------------------
## SECTION 5 — Important Implementation Details
------------------------------------------------------------

- **Performance considerations**: The pipeline is highly optimized. It uses a queue size of exactly 4 (`maxsize=4`) for the save thread to prevent memory bloat if the GPU processes images much faster than the hard drive can write them.
- **Thread safety**: The global deep learning model singleton is shielded by `threading.Lock()`, guaranteeing it won't crash if two separate Python threads try to initialize the pipeline simultaneously.
- **Caching behavior**: The cache validates based on the true SHA256 byte-content of the image. This is highly robust (renaming the image won't trigger a re-run) but does mean the script pays a heavy disk-read penalty just to generate the hash before the GPU even spins up.
- **Memory usage considerations**: The prefetch pool is limited strictly to `max_workers=1`. This is intentional; holding multiple high-res uncompressed OpenCV matricies in RAM simultaneously before they reach the GPU could cause OOM (Out Of Memory) system crashes.
- **Assumptions the code makes**: It assumes the underlying `base_meta_code.sam_3d_body` module is installed and functional. It assumes standard color spaces (OpenCV defaults to BGR, but it manually converts to RGB right before hitting the estimator).
- **Edge cases**: If `max_side <= 0`, downscaling is disabled. If an image has no people detected or detection confidence is too low, the generated `.npz` payload simply contains a completely empty array of people (handling empty arrays gracefully is left to the downstream user).

------------------------------------------------------------
## SECTION 6 — Risks / Pitfalls
------------------------------------------------------------

- **Where it could fail**: 
  - If the directory passed in `input_path` contains millions of images, `glob` memory expansion could stall the initial startup.
  - If a corrupted image crashes OpenCV internally rather than returning cleanly.
- **Hidden assumptions**: It assumes that the X and Y bounding box matrices are the only things that need scaling when `map_2d_to_original` is enabled. It does not scale 3D translation vectors or focal lengths because those exist in a different 3D focal space relative to the camera, which may confuse users attempting to project 3D coordinates onto 2D image planes manually.
- **Ambiguous behavior**: `pick_single_person` determines the "largest" person based purely on bounding box Area, not confidence metric. A blurry object close to the camera may override a perfectly clear human further away.
- **Things that should be documented but aren’t**:
  - The exact schema of the output `.npz` file and the shapes of the arrays it contains are largely undocumented in the `to_npz_payload` function. Downstream users will have a hard time knowing what `pred_vertices` actually looks like unless they inspect the NPZ file manually.
  - The `infer_ms` statistic only measures raw GPU time, omitting I/O read, resizing overhead, and async save overhead. Users may be misled regarding actual total throughput speed.
