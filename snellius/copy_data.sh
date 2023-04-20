mkdir temp
zip -r temp/preprocessed.zip data/preprocessed
scp temp/preprocessed.zip vloos@snellius.surf.nl:~/data/preprocessed.zip
rm -rf temp

ssh vloos@snellius.surf.nl "unzip preprocessed.zip"; "rm data/preprocessed.zip"