conda deactivate && \
conda env remove -n thann_maect -y && \
conda clean --all -y && \
conda env create -f environment_thann_20241013_part1.yml -y && \
conda clean -i -t -c -l -y && \
conda activate thann_maect && \
conda env update -f environment_thann_20241013_part2.yml && \
conda clean -i -t -c -l -y && \
sed -i '/from torch._six import string_classes/d' /system/apps/studentenv/thann/thann_maect/lib/python3.11/site-packages/pytorch_concurrent_dataloader/*.py && \
sed -i 's/string_classes/str/' /system/apps/studentenv/thann/thann_maect/lib/python3.11/site-packages/pytorch_concurrent_dataloader/*.py
