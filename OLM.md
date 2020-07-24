```
the variable formely known as N:
N = tileSize
```

tileChunk instead of tileSize allows for specifying a larger set of pixels to the smallest possible chunk that an image can be broken down in to.

chunkSize = pixel density of the a tile
tileSize = number of chunks in the side of a tile
stepSize?

1. Read in the image.
2. imageWidth % chunkSize * tileSize should eq 0 (same for height if not square)
3. Iterate through the image by in chunkSize * tileSize tiles in steps
4. Add overlapping relationships to vertices and store unique tile indexes
5. 

6. add orphan vertices