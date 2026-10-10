# EmoCodec datasets for Google Colab

## 1. Recommended starter — Oxford-IIIT Pet

- **Official page:** https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Torchvision API:** https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html
- **Size:** roughly 800 MB including images and annotations (Oxford's estimate).
- **Content:** ~7,400 photographs from 37 cat and dog breeds, varied pose, scale, lighting and backgrounds.
- **License:** Creative Commons Attribution-ShareAlike 4.0 International per Oxford. Copyright remains with original image owners; comply with attribution and license requirements.
- **Downloads:** the Colab notebook calls `torchvision.datasets.OxfordIIITPet(root=..., split="trainval", download=True)`. It downloads official assets and handles extraction.
- **Split:** seed 42 with 10% of the published *trainval* held out for validation; preserve the official *test* split untouched until final assessment. Notebook limits validation for faster epochs.
- **Caveat:** pets are a *narrow distribution*. Strong results on this data do not establish performance for landscapes, documents, UI screenshots or arbitrary photos. Do not train or benchmark general-purpose compression solely on this set.

## 2. Larger image compression research — DIV2K

- **Official page:** https://data.vision.ee.ethz.ch/cvl/DIV2K/
- **Training:** 800 high-resolution images.
- **Validation:** 100 high-resolution images.
- **Test:** 100; follow the official split and availability.
- **Content:** varied scenes, high resolution; supports multiple crop patches per image.
- **License:** **academic research purposes only**, according to the official ETH Zürich page. Not a safe default for commercial training without checking rights.
- **Resource consideration:** downloading the HR ZIP can take multiple GB of bandwidth and disk. Full HR frames are much larger than 256px patches.
- **Notebook workflow:** download the official DIV2K_train_HR.zip into Colab separately, extract it, set DATASET_CHOICE="local_folder" and LOCAL_IMAGES to the extracted image folder. For a meaningful validation protocol, keep DIV2K_valid_HR separate; the starter notebook's local_folder mode instead holds out part of the given training images. Do not mix official train and validation archives together.

Example Colab-only commands to run in an optional code cell (network availability not guaranteed):
```python
!wget -c -O /content/DIV2K_train_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
!unzip -q /content/DIV2K_train_HR.zip -d /content/
# Set DATASET_CHOICE="local_folder" and LOCAL_IMAGES="/content/DIV2K_train_HR"
```

## 3. Practical training guidance

1. For **pipeline smoke tests**, start with Oxford-IIIT Pet, 5 epochs, batch 8, images 128–256px.
2. For **better pet photos**, run 30+ epochs and inspect reconstruction/PSNR, codebook usage, and overfitting.
3. For **research-quality general images**, combine more diverse licensed photo sets, and keep separate validation/test sets.
4. Avoid comparing PSNR numbers from different datasets. Report exact training set and test set.
5. Include actual `.emotxt` UTF-8 bytes, `.emoc` bytes (header included), model weights size and consistent bit-per-pixel denominators.
6. Use JPEG, WebP and AVIF as *matched-rate* reference codecs before making compression superiority claims.

## 4. Checkpoint usage

The notebook produces `emocodec_best.pt` (compatible with desktop EmoCodec VQ-VAE). Copy the file into `checkpoints/photos_v1.pt`. Do not rename the model architecture, reassign emoji codebook indices, or resave checkpoints with different file bytes after encoding existing images. Their checkpoint SHA-256 IDs must match.

### References

Oxford: https://www.robots.ox.ac.uk/~vgg/data/pets/

Torchvision: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html

DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/

Colab resource limits: https://research.google.com/colaboratory/faq.html
