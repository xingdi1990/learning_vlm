# Download llava_v1_5_mix665k.json if it doesn't exist
if [ ! -f "llava_v1_5_mix665k.json" ]; then
    wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json
fi

# Download and unzip COCO train2017.zip if not already downloaded
if [ ! -f "train2017.zip" ]; then
    wget http://images.cocodataset.org/zips/train2017.zip
fi
unzip train2017.zip -d coco

# Download and unzip GQA images.zip if not already downloaded
if [ ! -f "gqa_images.zip" ]; then
    wget -O gqa_images.zip https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
fi
unzip gqa_images.zip -d gqa

# Download and unzip TextVQA train_val_images.zip if not already downloaded
if [ ! -f "train_val_images.zip" ]; then
    wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
fi
unzip train_val_images.zip -d textvqa

# Download and unzip VG_100K images.zip if not already downloaded
if [ ! -f "vg_images1.zip" ]; then
    wget -O vg_images1.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
fi
unzip vg_images1.zip -d vg


# Download and unzip VG_100K_2 images2.zip if not already downloaded
if [ ! -f "vg_images2.zip" ]; then
    wget -O vg_images2.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
fi
unzip vg_images2.zip -d vg


gdown https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing --folder

cd OCR-VQA-200K

if [ ! -f "ocr_vqa.tar" ]; then
    wget https://huggingface.co/datasets/ej2/llava-ocr-vqa/resolve/main/ocr_vqa.tar
fi

tar -xvf ocr_vqa.tar

mv ocr_vqa/images .

rm -rf ocr_vqa

cd ..

mv OCR-VQA-200K ocr_vqa

wget http://ecx.images-amazon.com/images/I/51YTH4k3fUL.jpg
cp 51YTH4k3fUL.jpg ocr_vqa/images/1437717772.jpg
rm 51YTH4k3fUL.jpg

