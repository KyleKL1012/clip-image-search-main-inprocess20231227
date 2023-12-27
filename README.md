## 1. Import images
There are two modes on importing images into database: `import` mode and `copy` mode.
`import` mode do not copy file, and you have to keep the original file later.
`copy` mode is useful when you need to scan temp or cache folder, and you can delete the original file after importing.
In both mode, the script will scan folder recursively and import all images files into database.

```
# import mode
python import_images.py /path/to/folder

# copy mode
python import_images.py /path/to/folder --copy
```

## 2. Start search engine 
```
python server.py
```


