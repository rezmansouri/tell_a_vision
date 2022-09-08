from gtts import gTTS


class Narrator:
    def __init__(self, class_labels: list, audio_directory: str, max_obj_per_segment=5,
                 rank_labels=('close', 'near', 'far'), h_direction_labels=('left', 'middle', 'right'),
                 v_direction_labels=('above', 'midst', 'bottom'), horizontal_only=True):
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
