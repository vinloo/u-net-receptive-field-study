scp train.py vloos@snellius.surf.nl:~/train.py
scp unet.py vloos@snellius.surf.nl:~/unet.py
scp utils/config.py vloos@snellius.surf.nl:~/utils/config.py
scp preprocess_data.py vloos@snellius.surf.nl:~/preprocess_data.py

mkdir temp
zip -r temp/configurations.zip configurations
scp temp/configurations.zip vloos@snellius.surf.nl:~/configurations.zip
rm -rf temp

ssh vloos@snellius.surf.nl "unzip configurations.zip"; "rm configurations.zip"
