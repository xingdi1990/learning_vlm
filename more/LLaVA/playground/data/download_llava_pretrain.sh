mkdir -p LLaVA-Pretrain

cd LLaVA-Pretrain

wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json

wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k_meta.json

wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip

unzip images.zip

rm images.zip

cd ..

