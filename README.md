# misc

# image_collector

Collect images from internet and detect object (human frontal face) in the images.
Image collection is thanks to GoogleImageDownloader. Image detection is thanks to OpenCV.

usage:
```
$ ./ImageCollector.py 
usage: ImageCollector [-h] -k KEYWORDS [-d DEST] [-s] [-o]
                      [-t SEARCHIMAGETYPE]
                      [-l {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100}]
```

 - *-k, --keywords* specify search keywords split by comma. e.g. "白鵬,角竜,稀勢の里"
 - *-o, --detectObject* detect object (human frantal face) in the images.
 - *-s, --skipDownload* skip download. -o should be specified instead.
 - *-d* images destination. if -o (--detectObject) specified, the detected objects are saved in "objects" subdirectory.
 - *-l, --downloadLimit* image download limit. 1 to 100.
 - *-t, --searchImageType* google image search type: 'face','photo','clip-art','line-drawing','animated'

# scripts

some scripts comes from tutorials and examples.
