# Obtain static gesture

### useful links

https://google.github.io/mediapipe/solutions/hands.html

https://www.hackster.io/as4527/volume-control-using-hand-gesture-using-python-and-opencv-7aab9f


## Running the static gesture recognizer Example (see also get-image example in sdk)
This example is used to create popup windows which show a live preview of the image sources specified.

Additionally, the `mediapipe` module is used to extract hand landmarks that are used to interpret hand gestures.
 
To run the example and display images from the base robot cameras:
```
python3 obtain_static_gesture.py --image-sources frontleft_fisheye_image
```

The command specifies each source from which images should be captured using the command line argument `--image-sources`.
Additionally, the arguments `--image-service` and `--auto-rotate` are can be used. For details see the get-image example in the spot sdk.

The argument `--jpeg-quality-percent` can be provided to change the JPEG quality of the requested images; this argument describes a percentage (0-100) for the quality.

The argument `--capture-delay` can be used to change the wait time between image captures in milliseconds.

If only a single image source is requested to be displayed, by default the program will make the image viewer show a full screen stream. To disable this, provide the argument `--disable-full-screen`, which will make the image stream display auto-size to approximately the size of the image.

