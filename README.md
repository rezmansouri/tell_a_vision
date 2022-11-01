# tell a vision 📺

<p align="center">
  <img src="https://github.com/rezmansouri/tell_a_vision/blob/main/misc/tv.png" width="100%"/>
</p>

**tell a vision (`tv`)**  is an inference engine, in the form of a Python package that describes the scenes for object detection tasks in computer vision.

## Links
- [tell-a-vision on PyPI](https://pypi.org/project/tell-a-vision/)
- [Demo Colab Notebook](https://colab.research.google.com/drive/1o6MgntmIb1qLLpGXxsDw0W6SMH2kheCZ?usp=sharing)



## Introduction
The task of object detection consists of three subtasks: object recognition, localization, and classification. Usually, object detection projects present their output in a visual format that shows the objects found with bounding boxes in different colors representing their classes. Something like this:

<p align="center">
  <img src="https://github.com/rezmansouri/tell_a_vision/blob/main/misc/dog-bike-truck.png" width="400em"/>
</p>

But what if we could make the computer tell us what it is seeing? For example: "There is a dog in the bottom left, a bicycle in the middle, and a truck on the top right." What if the computer can *tell a vision*? Pretty cool right?

**tell a vision (`tv`)** is a Python package that can provide explanatory analysis on the output of object detection algorithms. It takes bounding boxes, and the classes of the objects found and answers questions such as:

- How many objects and of what kind are in a specific region of the scene?
- How far are they? Are they close? 
- Are they small or medium-sized?
- ...

And in the end, using TTS (Text To Speech) it can describe the scene.

Here is a mere representation of what **`tv`** does:

<p align="center">
  <img src="https://github.com/rezmansouri/tell_a_vision/blob/main/misc/tv.gif" width="100%"/>
</p>

You can try [this](https://colab.research.google.com/drive/1o6MgntmIb1qLLpGXxsDw0W6SMH2kheCZ?usp=sharing) colab notebook as a demo.

## Reference
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
  <img src="https://github.com/rezmansouri/tell_a_vision/blob/main/misc/car-pov-2.png" width="100%"/>
</p>

The corresponding `boxes` and `classes` for this visualization, provided by the object detection algorithm, are:
```python
boxes = [[275,    4,  546,  400],
         [290,  350,  371,  431]
         [294,  425,  395,  540],
         [268,  521,  357,  597],
         [288,  610,  390,  717],
         [250,  638,  272,  660]
         [285,  733,  356,  787],
         [145,  775,  427,  987],
         [252,  810,  422,  867],
         [266,  971,  370, 1013],
         [270, 1018,  355, 1056],
         [104, 1133,  155, 1186],
         [163, 1141,  211, 1184]]

classes = [1, 1, 1, 2, 3, 1, 1, 2, 0, 0, 0, 4, 4]
```
This scene (picture) is 1280px wide and 720px high. The objects are mentioned from left to right (if you're following the coordinates) and the class labels are:
```python
CLASS_LABELS = ['pedestrian', 'car', 'truck', 'traffic light', 'traffic sign']
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
 [0, 0],
 [0, 0],
 [0, 0],
 [1, 0],
 [2, 0],
 [2, 0],
 [2, 1],
 [2, 0],
 [2, 0]]
```

#### Task two: Distance estimation with `tv.Ruler`
`tv.Ruler` needs to be fit on a dataset, preferably big enough, to obtain quartiles of each object class's areas. By doing so it will be able to place a new object into these quartiles according to its pixel area, as a measure of its size/distance in the scene.

Suppose we have a dataset of 1000 images with their corresponding bounding box annotations in a list called `images`. The list would look something like this:
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
            'class': 'truck'
        },
        {
            'box': {
                    'x1': 90.107,
                    'y1': 467.11,
                    'x2': 100.86,
                    'y2': 500.42
                   },
            'class': 'pedestrian'
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
            'class': 'truck'
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
        {
            'box': {
                    'x1': 80.107,
                    'y1': 367.11,
                    'x2': 90.83,
                    'y2': 450.4
                   },
            'class': 'traffic light'
        },
        {
            'box': {
                    'x1': 90.7,
                    'y1': 467.11,
                    'x2': 95.86,
                    'y2': 510.98
                   },
            'class': 'traffic sign'
        },
    ],
    ...
]
 
```
Each element in `images` represents the annotation of an image in your dataset and contains bounding box coordinates along with classes of the objects in the image.

```python
ruler = tv.Ruler(images=images, classe_labels=CLASS_LABELS) # Fitting the ruler on your annotations
```

After, although a private variable and inaccessible, ruler's quartiles would be something like this:
```python
{
    'truck':         [200.12,  405.85,  793.11],
    'car':           [536.11,  863.94,  1076.2],
    'pedestrian':    [12.133,  60.353,  100.34],
    'traffic light': [5.65,    18.46,    52.11],
    'traffic sign':  [4.42,    15.96,    89.91]
}
```
Now with `ruler.get_ranks()` we are able to find the quartile intervals of the objects we've found:
```python
ranks = ruler.get_ranks(boxes=boxes, classes=classes)
```
Each element of `ranks` corresponds to the found object's specific class quartile interval (0, 1, 2, or 3). `ranks` would be of shape `(n,)` and look something like this:
```python
[3, 2, 2, 2, 2, 0, 1, 3, 3, 1, 1, 3, 3]
```
`0` can be inferred as small / far, `1` as relatively small / far, `2` as roughly big / near, `3` as big / near.

*As can be seen, moving from left to right, the cars are close, near, near, the truck is near, the car in the middle is near, the traffic light is far, the car parked on the right is near, the truck on the right and the pedestrian beside it are close, the two pedestrians on the right are near, and the two traffic signs are near.*

#### Task three: Scene narration with `tv.Narrator`
Now our goal is to group together our findings to get a concise description of the scene, and maybe let **`tv`** *finally tell the vision!*
```python
narrator = tv.Narrator(class_labels=CLASS_LABELS, audio_directory='./audio', horizontal_only=False)
```
This will create a narrator object and download all possible narrations for your classes as `.mp3` files in `./audio`. The directory will look something like this:
- 1-car-left-bottom-near.mp3
- 2-car-left-midst-near.mp3
- ...
- 4-traffic light-middle-above-far.mp3
- ...
- 5-pedestrian-middle-above-close.mp3

Finally, using `narrator.get_narration()` static method we can get the summarized description of the scene in the format of the name of the audio files to be played:
```python
narration = narrator.get_narration(classes=classes, class_labels=CLASS_LABELS, ranks=ranks, locations=locations)
```
`narration` for our example would be something like this:
```python
[
    '1-car-left-bottom-close',
    '2-car-left-above-near',
    '1-truck-left-above-near',
    '1-car-middle-above-near',
    '1-traffic light-right-above-far',
    '1-car-right-above-near',
    '1-truck-right-above-close',
    '1-pedestrian-right-above-close',
    '2-pedestrian-right-above-near',
    '2-traffic sign-right-above-near'
 ]
```
If since the beginning of this procedure, `horizontal_only` was set to `True` (default value) everywhere, the resulting `narration` would've been like this:
```python
[
    '1-car-left-close',
    '2-car-left-near',
    '1-truck-left-near',
    '1-car-middle-near',
    '1-traffic light-right-far',
    '1-car-right-near',
    '1-truck-right-close',
    '1-pedestrian-right-close',
    '2-pedestrian-right-near',
    '2-traffic sign-right-near'
 ]
```
Specific for you own application, you may choose to play these audio files from `./audio` consecutively, low-speed, high-speed, merge them together, etc.. Not **`tv`**'s business! But lets hear a demo:
```python
import IPython
for nar in narration:
  IPython.display.display(IPython.display.Audio(f'audio/{nar}.mp3'))
```

https://user-images.githubusercontent.com/46050829/189340077-6022a955-d48d-4ba2-9f57-4fd00d3fa627.mp4

Lets take a look at the picture again:

<p align="center">
  <img src="https://github.com/rezmansouri/tell_a_vision/blob/main/misc/car-pov-3.png" width="100%"/>
</p>

### Documentation
#### `tv.locate(boxes, scene_width, scene_height, v_point=.66, h_point=.66, horizontal_only=True)`
#### Arguments
- `boxes`: numpy array of shape (n, 4) where n is the number of the boxes found and each element contains bounding box of the found object in the following format: `[ymin, xmin, ymax, xmax]`.
- `scene_width`: integer representing the width of the scene in pixels.
- `scene_height`: integer representing the height of the scene in pixels.
- `v_point`<sup>1</sup>: portion of an objects height to be considered as a threshold when it is placed vertically near-midst to determine its vertical placement. Defaults to `0.66`.
- `h_point`<sup>1</sup>: portion of an objects width to be considered as a threshold when it is placed horizontally near-middle to determine its horizontal placement. Defaults to `0.66`.
- `horizontal_only`: Whether to locate the objects only horizontally. Defaults to `True`.
- `returns` array of shape (n, 2) containing the locations of the boxes: `[h_location, v_location]`. `0`, `1`, and `2` mean left/above, middle/midst, right/bottom for `h_location` and `v_location` respectively. if `horizontal_only=True`, `v_location` will be `-1` for all entries.

1- `v_point` and `h_point` are described more in detail here:

<p align="center">
  <img src="https://github.com/rezmansouri/tell_a_vision/blob/main/misc/vpoint-hpoint.png" width="80%"/>
</p>

Using these arguments, you can specify the portion of an object's width/height that if laid on one side of the h_margin/v_margin here, the object's placement be considered as that side's direction.

In other words:

- if $\alpha$ > `h_point` * object's width, it is considered on the right.
- if $\beta$ > `h_point` * object's width, it is considered on the left.
- otherwise, it is considered in the middle.

Vertically,
- if $\rho$ > `v_point` * object's height, it is considered above.
- if $\sigma$ > `v_point` * object's height, it is considered in the bottom.
- otherwise, it is considered in the midst.

#### `tv.Ruler(images, class_labels, coords_key='box', class_key='class', xmin_key='x1', ymin_key='y1', xmax_key='x2', ymax_key='y2')`
A `Ruler` object learns the range of each object class size from a dataset and can provide size/distance estimation for new objects found by the object detection algorithm.
#### Arguments
- `images`: list of image annotations, where each image annotation contains object annotations (bounding box coordinates and class). i.e. this is an acceptable input:
```python
[
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
            'class': 'bike'
        },
    ],
    [
        {
            'box': {
                    'x1': 90.107,
                    'y1': 467.11,
                    'x2': 100.86,
                    'y2': 500.42
                   },
            'class': 'person'
        },
        {
            'box': {
                    'x1': 867.12,
                    'y1': 716.50,
                    'x2': 1250.9764,
                    'y2': 987.42
                   },
            'class': 'truck'
        },
        ...
    ],
    ...
]
```
- `class_labels`: list of class labels of objects in your dataset. For example: `['car', 'bike', 'person', 'truck']`.
- `coords_key`: alternative key for `'box'` in image annotations.
- `class_key`: alternative key for `'class'` in image annotations.
- `xmin_key`: alternative key for `'x1'` in image annotations.
- `ymin_key`: alternative key for `'y1'` in image annotations.
- `xmax_key`: alternative key for `'x2'` in image annotations.
- `ymax_key`: alternative key for `'y2'` in image annotations.
- `returns` a `Ruler` object that can be later used to determine detected objects' size/distance.

#### `tv.Ruler.get_rank(boxes, classes)`
Returns the quartile interval index of `boxes` according to `classes` and the range learned specific to each class label while the `ruler` was initialized. 
#### Arguments
- `boxes`: array of shape (n, 4) with each element containing `[ymin, xmin, ymax, xmax]` coordinates of bounding boxes.
- `classes`: array of shape (n, ) with each element corresponding to the index of the object's class in `ruler`'s class labels.
- `returns` array of shape (n, ) with each element being either 0, 1, 2, or 3, interpretable as small/far, relatively small/far, relatively close/big, close/big respectively for each box.

<p align="center">
  <img src="https://github.com/rezmansouri/tell_a_vision/blob/main/misc/quartiles.png" width="80%"/>
</p>

#### `tv.Narrator(class_labels, audio_directory, max_obj_per_segment=5, rank_labels=('close', 'near', 'far'), h_direction_labels=('left', 'middle', 'right'), v_direction_labels=('above', 'midst', 'bottom'), horizontal_only=True)`
By creating a `Narrator`, all possible audio narrations for `class_labels` will be downloaded to `audio_directory` and you'll later be able to use the `Narrator` to summarize a scene's findings by **`tv`** into narrations.
#### Arguments
- `class_labels`: list of class labels of objects in your dataset. For example: `['car', 'bike', 'person', 'truck']`.
- `audio_directory`: string path of a directory to save downloaded `.mp3` narrations to. If it doesn't exist, **`tv`** will attempt to create it.
- `max_obj_per_segment`: maximum number of objects that are summarized together. For example, the number of cars on the right side of the scene that are close. Defaults to `5`.
- `rank_labels`: labels associated with the ranks of the objects. Tuple of length more than zero and less than five to cover the possible quartiles in ranks. Defaults to `('close', 'near', 'far')`.
- `h_direction_labels`: labels associated with the horizontal location of the objects. Tuple of length three. Defaults to `('left', 'middle', 'right')`.
- `v_direction_labels`: labels associated with the vertical location of the objects. Tuple of length three. Defaults to `('above', 'midst', 'bottom')`.
- `horizontal_only`: whether to download annotations regarding only the horizontal location of the objects or not. Defaults to `True`.
- `returns` a `Narrator` object that can be later used to summarize the scene in narrations.

#### `tv.Narrator.get_narration(classes, class_labels, ranks, locations, rank_to_distance_labels=('far', 'near', 'near', 'close'), h_location_to_lr_labels=('left', 'middle', 'right'), v_location_to_ab_labels=('above', 'midst', 'bottom'), horizontal_only=True)`
A static method that receives the output of `tv.locate()`, and `tv.Ruler.get_ranks()` and returns the summary of the scene in the form of narrations.

- `classes`: array of shape (n, ) with each element corresponding to the index of the object's class in `class_labels`.
- `class_labels`: list of class labels of objects in your dataset. For example: `['car', 'bike', 'person', 'truck']`.
- `ranks`: output of `tv.Ruler.get_rank()`. Array of shape (n, ) with each element being 0, 1, 2, or 3 representing an object's size/distance.
- `locations`: output of `tv.locate()`. Array of shape (n, 2) containing the locations of the boxes: `[h_location, v_location]`.
- `rank_to_distance_labels`: tuple of length 4 describing objects' distance/size according to their ranks (quartile intervals). Defaults to `('far', 'near', 'near', 'close')`.
- `h_location_to_lr_labels`: tuple of length 3 describing objects' horizontal location. Defaults to `('left', 'middle', 'right')`.
- `v_location_to_ab_labels`: tuple of length 3 describing objects' vertical location. Defaults to `('above', 'midst', 'bottom')`.
- `horizontal_only`: whether to create narrations regarding objects' vertical location. Must be set to `True` if a `narrator`'s audio files were created with it set true in `tv.Narrator()`. Defaults to `True`.
- `returns` a list of narrations that correspond to the downloaded audio files for a `narrator` which can also be used in a textual format.

## Contributions
The idea of developing **`tv`** originated when I was working on my B.Sc. project. There might be issues with it, but it can surely be improved. Pull requests are welcome and you can reach my at my [email](mailto:std_reza_mansouri@khu.ac.ir) if you need to discuss something or become a collaborator.
