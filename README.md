# OpenCV-Color-Segmentation
Research code on methods for segmentation of foreground image from green screen background.

----
## Segmentation
Segmentation involves extracting specific parts of the image to make the image simpler or 
to extract a region of interest like a foreground object from the image. 

This repsitory consists of general code that was used for foreground and background segmentation for the specific use case of images taken in a greenscreen under random lighting conditions. The general name of this technique is called as [chroma-key](https://en.wikipedia.org/wiki/Chroma_key) segmentation.

The general methods of segmentation can be broadly classified into :
1. Classical computer-vision methods : Graphcut techniques , using color based segmentations 

1. Deep Learning methods : U-Net for image segmentation etc.


----
# References 
1. https://github.com/andrewssobral/bgslibrary/wiki/List-of-available-algorithms
2. http://www.cs.utah.edu/~michael/chroma/
3. http://gc-films.com/chromakey.html
