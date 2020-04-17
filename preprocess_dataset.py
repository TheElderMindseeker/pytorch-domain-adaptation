import os
import tarfile
import tempfile

from PIL import Image
from tqdm import tqdm

source = tarfile.open('./small_dataset.tar.gz', 'r:gz')
target = tarfile.open('./optimized_pfe_dataset.tar.gz', 'w:gz')

_, image_temp = tempfile.mkstemp('.jpg')

for name in tqdm(source.getnames()):
    file_data = source.extractfile(name)
    image_data = Image.open(file_data, 'r')
    image_data = image_data.resize((398, 224))
    image_data.save(image_temp)
    target.add(image_temp, name)

os.unlink(image_temp)
