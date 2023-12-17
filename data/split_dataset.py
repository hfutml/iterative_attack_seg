
import os
import json

data_path = "./data_res10.json"

def get_test_story_context(test_path):
    data = json.load(open(test_path, 'r'))
    for x in data:
        print(x)
def get_story_context(story_json_path, mode):
    story_data = json.load(open(story_json_path))
    story_info = {}
    story_info['src_word_to_ix'] = story_data['src_word_to_ix']
    story_info['src_ix_to_word'] = story_data['src_ix_to_word']
    story_info['tgt_word_to_ix'] = story_data['tgt_word_to_ix']
    story_info['tgt_ix_to_word'] = story_data['tgt_ix_to_word']
    images = []
    for i in range(len(story_data['images'])):
        if story_data['images'][i]['split'] == mode:
            images.append(story_data['images'][i])
    story_info['images'] = images

    return story_info

# story = get_story_context(data_path, 'val')
#
# with open('val_data_res10.json', 'w') as ff:
#     json.dump(story, ff)
# # json.dump(story, open('story.json', 'w'))
# ff.close()

test_path = 'annotation.json'
get_test_story_context(test_path)