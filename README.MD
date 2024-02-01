# Few shot classification with DINOv2

This is a repository to show how DINOv2 can be used to perform few shot classification.
Install python requirements with `pip install -r requirements.txt`.

Download the images and / or the features with
```bash
wget https://partage.imt.fr/index.php/s/3pJAnmkwrR7pntG/download -O images.zip
unzip images.zip
wget https://partage.imt.fr/index.php/s/rGfacYEyAK2ZXNz/download -O features.zip
unzip features.zip
```

Launch the script with `python main.py --nb-shots 5 --nb-queries 2`.

You can then hover on the images to see the images and how they've been classified.
