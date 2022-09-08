# tell a vision 📺

<p align="center">
  <img src="https://github.com/rezmansouri/tell_a_vision/blob/main/misc/tv.png" width="400em"/>
</p>

## Introduction
The task of object detection consists of three subtasks: object recognition, localization, and classification. Usually, object detection projects present their output in a visual format that shows the objects found with bounding boxes in different colors representing their classes. Something like this:

<p align="center">
  <img src="https://github.com/rezmansouri/tell_a_vision/blob/main/misc/dog-bike-truck.png" width="400em"/>
</p>

But what if we could make the computer tell us what it is seeing? For example: "There is a dog in the bottom left, a bicycle in the middle, and a truck on the top right." What if the computer can *tell a vision*? Pretty cool right?

**tell a vision (`tv`)** is Python package that can provide explanatory analysis on the output of object detection algoirthms. It takes bounding boxes, and the classes of the objects found and answers questions such as:

- How many objects and of what kind are in a specific region of the scene?
- How far are they? Are they close? 
- Are they small or medium-sized?
- ...

And in the end, using TTS (Text To Speech) it can describe the scene.

Here is a mere representaion of what **`tv`** does:

<p align="center">
  <img src="https://github.com/rezmansouri/tell_a_vision/blob/main/misc/tv.gif" width="700em"/>
</p>

## Documentation
Try **`tv`** using [this]() demo notebook on your local machine or via colab [here]().
### Requirements
1. Python interpreter version 3.7 or later
2. Package prerequisites:
    - numpy version 1.21 or later
    - gTTs (Google Text to Speech) version 2.2.4 or later

Note: *Package prerequisites will be checked and installed if you'll follow the installation step, otherwise if you're using the source directly, you'll need to check and install them yourself.*
### Installation
You can install **`tv`** via `pip` or by downloading the source and installing it manually.
#### Method one:
**`tv`** is available on PyPI and installable with `pip`:
```
pip install --upgrade tell-a-vision
```

#### Method two:
1- Download this repository as zip and extract it or use:
```
git clone https://github.com/rezmansouri/tell_a_vision.git
```
2- Change directory to the package's root:
```
cd tell_a_vision
```
3- Make sure you have setuptools package installed:
```
pip install --upgrade setuptools
```
4- Install the package via the `setup.py ` script:
```
python setup.py install
```

### Basic usage
**`tv`** can do more than what you need, but its complete goal is as follows: First, to tell the placement of the objects your algorithm had found (i.e. right, left, middle, above, bottom, midst). `tv.locate()` does this task of describing the localization. Further, `tv.Ruler` does the task of describing the size / distance of the objects found (i.e. close, near, far, or small, medium, big). The last but not least feature of this package, is `tv.Narrator` where it can take the output of the two former features and describe the scene fully via text or audio.

Normally your object detection algorithm will output these results:

1. **`boxes`**: an array (tensor) of shape `(n, 4)` where `n` is the number of boxes found and the last axis contains `[ymin, xmin, ymax, xmax]` of the found object's bounding box coordinates.

2. **`classes`**: an array (tensor) of shape `(n,)` where `n` is the number of boxes found and it contains the integer index of the found objects' class labels.

3. Other outputs: confidence score, class confidence score, and other outputs that **`tv`** won't have anything to do with.

**Note**: **`tv`** works on the final output of your object detection algorithm. It neither does *non-max suppression* nor rescaling the outputs. So make sure the inputs your providing are the ones that you consider as the final results.

For the rest of this documentation, let's consider the following output of an object detection algorithm, such as YOLO, as an example:

<p align="center">
  <img src="https://github.com/rezmansouri/tell_a_vision/blob/main/misc/street-2.jpg" width="700em"/>
</p>

The corresponding `boxes` and `classes` for this visualization, provided by the object detection algorithm, are:
```python
boxes = [[414,    0,  560,  163],
         [370,  167,  559,  242],
         [390,  175,  584,  603],
         [155,  316,  208,  403],
         [337,  337,  387,  413],
         [420,  293,  712, 1051],
         [375,  912,  414,  963],
         [400, 1065,  603, 1140]]

classes = [1, 0, 1, 2, 2, 1, 2, 0]
```
This scene (picture) is 1280px wide and 720px high. The objects are mentioned from left to right (if you're following the coordinates) and the class labels are:
```python
CLASS_LABELS = ['person', 'car', 'advertisement']
```

#### Task one: Localization with `tv.locate()`
First, let's see if the objects we've found are placed left, middle, right (or optionally, above, midst, bottom) of the scene.
```python
import tell_a_vision as tv

locations = tv.locate(boxes=boxes, scene_width=1280, scene_height=720, horizontal_only=False)
```
`tv.locate` returns an array of shape `(n, 2)` where the last axis's first element represent horizontal location (0: left, 1: middle, 2: right), and the second element represents the vertical location (0: above, 1: midst, 2: bottom) of the nth object. 


The `locations` result for our example would be:
```python
[[0, 2],
 [0, 2],
 [0, 2],
 [0, 0],
 [0, 1],
 [1, 2],
 [2, 2],
 [2, 2]]
```

#### Task two: Distance estimation with `tv.Ruler`
`tv.Ruler` needs to be fit on a dataset, preferably big enough, to obtain quartiles of each object class's areas. By doing so it will be able to place a new object into these quartiles according to its pixel area, as a measure of its size/distance in the scene.

Suppose we have a dataset of 1000 images with their corresponding bounding box annotations in a list call `images`. The list would look something like this:
```python
images = [
    [
        {
            'box': {
                    'x1': 400.12,
                    'y1': 700.50,
                    'x2': 1156.97,
                    'y2': 900.20
                   },
            'class': 'car'
        },
        {
            'box': {
                    'x1': 867.12,
                    'y1': 716.50,
                    'x2': 1250.9764,
                    'y2': 987.42
                   },
            'class': 'car'
        },
        {
            'box': {
                    'x1': 90.107,
                    'y1': 467.11,
                    'x2': 100.86,
                    'y2': 500.42
                   },
            'class': 'advertisement'
        },
    ],
    [
        {
            'box': {
                    'x1': 665.44,
                    'y1': 133.18,
                    'x2': 688.556,
                    'y2': 210.87
                   },
            'class': 'person'
        },
        {
            'box': {
                    'x1': 967.12,
                    'y1': 716.50,
                    'x2': 1254.964,
                    'y2': 987.42
                   },
            'class': 'car'
        },
    ],
    ...
]
 
```
Each element in `images` represents the annotation of an image in your dataset and contains bounding box coordinates along with classes of the objects in the image.

```python
ruler = tv.Ruler(images=images, classes=CLASS_LABELS) # Fitting the ruler on your annotations
```

After, although a private variable and inaccessible, ruler's quartiles would be something like this:
```python
{
    'person':        [200.12,  405.85,  793.11],
    'car':           [536.11,  863.94,  1076.2],
    'advertisement': [12.133,  60.353,  100.34]
}
```
Now with `ruler.get_ranks()` we are able to find the quartile intervals of the objects we've found:
```python
ranks = ruler.get_ranks(boxes=boxes, classes=classes)
```
Each element of `ranks` corresponds to the found object's specific class quartile interval (0, 1, 2, or 3). `ranks` would be of shape `(n,)` and look something like this:
```python
[2, 2, 1, 2, 1, 1, 3, 0, 1]
```
`0` can be inferred as small / far, `1` as relatively small / far, `2` as roughly big / near, `3` as big / near.

#### Task three: Scene narration with `tv.Narrator`
Now our goal is to group together our findings to get a concise description of the scene, and maybe let **`tv`** *finally tell the vision!*
```python
narrator = tv.Narrator(classes_labels=CLASS_LABELS, audio_directory='./audio', horizontal_only=False)
```
This will create a narrator object and download all possible narrations for your classes as `.mp3` files in `./audio`. The directory will look something like this:
- 1-car-left-bottom-near.mp3
- 2-advertisement-right-midst-far.mp3
- ...
- 5-persons-middle-above-close.mp3

Finally, using `narrator.get_narration()` static method we can get the summerized description of the scene in the format of the name of the audio files to be played:
```python
narration = narrator.get_narration(classes=classes, class_labels=CLASS_LABELS, ranks=ranks, locations=locations)
```
`narration` for our example would be something like this:
```python
[
    '2-car-left-bottom-near',
    '1-person-left-bottom-far', 
    '1-advertisement-left-above-close', 
    '1-advertisement-left-midst-close',
    '1-car-middle-bottom-close', 
    '1-advertisement-right-bottom-far', 
    '1-person-right-bottom-near'
 ]
```
Specific for you own application, you may choose to play these audio files from `./audio` consecutively, low-speed, high-speed, merge them together, etc.. Not **`tv`**'s business! But lets hear a demo:
```python
import IPython
