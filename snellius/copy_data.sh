mkdir temp
zip -r temp/data.zip data
scp temp/data.zip vloos@snellius.surf.nl:~/data.zip
rm -rf temp

ssh vloos@snellius.surf.nl "unzip data.zip"; "rm data.zip"