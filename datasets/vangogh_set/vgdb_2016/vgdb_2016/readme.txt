This is the dataset VGDB-2016 built for the paper
"From Impressionism to Expressionism: Automatically
Identifying Van Gogh's Paintings", which will be
published on the 23rd IEEE International Conference
on Image Processing (ICIP 2016).


## Images description
- Folder "train" contains paintings used for training
- Folder "test" contains paintings used for testing
- Folder "check" contains the two paintings whose
    autorship is still under debate

Paintings image files are named by their corresponding
unique "Page ID" identifier in Wikimedia Commons.
Additionally, paintings by van Gogh are prefixed
with "vg", while the other paintings are prefixed
with "nvg".


## CSV description
- PageID: unique "Page ID" identifier in
    Wikimedia Commons
- DescriptionURL: URL to the description page of
    the image in Wikimedia Commons
- ImageURL: URL to the image
- ImageSHA1: SHA-1 hash of the image
- PixelHeight: Image height in pixels
- PixelWidth: Image width in pixels
- PaintingID: Unique painting identifier
    according to a catalogue
- Artist: Name of the author
- RealHeightInches: Painting height in inches
- RealWidthInches: Painting width in inches
- DensityHeight: Density in pixels per inch
    for height orientation
- DensityWidth: Density in pixels per inch
    for width orientation
- DensityRatio: Density ratio according to
    Equation (1) in the paper


## References
The paper is available at IEEE Xplore:
- https://dx.doi.org/10.1109/icip.2016.7532335

The dataset is available at figshare:
- https://dx.doi.org/10.6084/m9.figshare.3370627

The source code is available at GitHub:
- https://github.com/gfolego/vangogh


## Corresponding author
Anderson Rocha (anderson.rocha@ic.unicamp.br)


If you find this work useful in your research,
please cite the paper! :-)

@InProceedings{folego2016vangogh,
    author = {Guilherme Folego and Otavio Gomes and Anderson Rocha},
    booktitle = {2016 IEEE International Conference on Image Processing (ICIP)},
    title = {From Impressionism to Expressionism: Automatically Identifying Van Gogh's Paintings},
    year = {2016},
    month = {Sept},
    pages = {141--145},
    keywords = {Art;Feature extraction;Painting;Support vector machines;Testing;Training;Visualization;CNN-based authorship attribution;Painter attribution;Data-driven painting characterization},
    doi = {10.1109/icip.2016.7532335}
}
