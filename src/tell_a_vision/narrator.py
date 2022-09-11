from gtts import gTTS
import os


class Narrator:
    def __init__(self, class_labels: list, audio_directory: str, max_obj_per_segment=5,
                 rank_labels=('close', 'near', 'far'), h_direction_labels=('left', 'middle', 'right'),
                 v_direction_labels=('above', 'midst', 'bottom'), horizontal_only=True):
        """
        :param class_labels: list of class labels
        :param audio_directory: path of desired directory for audio narrations to be downloaded. if not exists, it will be created
        :param max_obj_per_segment: the maximum number of objects in a single narration
        :param rank_labels: labels associated with the ranks of the objects. Tuple of length more than zero and less than five to cover the possible quartiles in ranks
        :param h_direction_labels: labels associated with the horizontal location of the objects. Tuple of length three
        :param v_direction_labels: labels associated with the vertical location of the objects. Tuple of length three
        :param horizontal_only: whether to download annotations regarding only the horizontal location of the objects or not
        """
        if not os.path.isdir(audio_directory):
            os.mkdir(audio_directory)
        self._audio_directory = audio_directory
        self._horizontal_only = horizontal_only
        if not audio_directory.endswith(('/', '\\')):
            audio_directory += '/'
        for c in class_labels:
            if horizontal_only:
                for distance in rank_labels:
                    for direction in h_direction_labels:
                        text = f'one {c} {direction}-{distance}'
                        tts = gTTS(text, lang='en')
                        tts.save(f'{audio_directory}1-{c}-{direction}-{distance}.mp3')
            else:
                for distance in rank_labels:
                    for v_direction in v_direction_labels:
                        for h_direction in h_direction_labels:
                            text = f'one {c} {h_direction} {v_direction} {distance}'
                            tts = gTTS(text, lang='en')
                            tts.save(f'{audio_directory}1-{c}-{h_direction}-{v_direction}-{distance}.mp3')
            for i in range(2, max_obj_per_segment + 1):
                if horizontal_only:
                    for distance in rank_labels:
                        for direction in h_direction_labels:
                            text = f'{i} {c}s  {direction} {distance}'
                            tts = gTTS(text, lang='en')
                            tts.save(f'{audio_directory}{i}-{c}-{direction}-{distance}.mp3')
                else:
                    for distance in rank_labels:
                        for v_direction in v_direction_labels:
                            for h_direction in h_direction_labels:
                                text = f'{i} {c}s  {h_direction} {v_direction} {distance}'
                                tts = gTTS(text, lang='en')
                                tts.save(f'{audio_directory}{i}-{c}-{h_direction}-{v_direction}-{distance}.mp3')

    @staticmethod
    def get_narration(classes, class_labels, ranks, locations, rank_to_distance_labels=('far', 'near', 'near', 'close'),
                      h_location_to_lr_labels=('left', 'middle', 'right'),
                      v_location_to_ab_labels=('above', 'midst', 'bottom'),
                      horizontal_only=True):
        """
        :param classes: array of shape (n, ) with each element corresponding to the index of the object's class in class_labels
        :param class_labels: list of class labels of objects in your dataset. For example: ['car', 'bike', 'person', 'truck'].
ranks: output of tv.Ruler.get_rank(). Array of shape (n, ) with each element being 0, 1, 2, or 3 representing an object's size/distance
        :param ranks: output of tv.locate(). Array of shape (n, 2) containing the locations of the boxes: [h_location, v_location]
        :param locations: output of tv.locate(). Array of shape (n, 2) containing the locations of the boxes: [h_location, v_location]
        :param rank_to_distance_labels: tuple of length 4 describing objects' distance/size according to their ranks (quartile intervals)
        :param h_location_to_lr_labels: tuple of length 3 describing objects' horizontal location
        :param v_location_to_ab_labels: tuple of length 3 describing objects' vertical location
        :param horizontal_only: whether to create narrations regarding objects' vertical location. Must be set to True if a narrator's audio files were created with it set true in tv.Narrator()
        :return: a list of narrations that correspond to the downloaded audio files for a narrator which can also be used in a textual format
        """
        map_ = {}
        for i, c in enumerate(classes):
            label = class_labels[c]
            key = f'-{label}-{h_location_to_lr_labels[locations[i][0]]}'
            if not horizontal_only:
                key += f'-{v_location_to_ab_labels[locations[i][1]]}'
            key += f'-{rank_to_distance_labels[ranks[i]]}'
            if key in map_:
                map_[key] += 1
            else:
                map_[key] = 1
        narration = []
        for key in map_:
            narration.append(f'{map_[key]}{key}')
        return narration
