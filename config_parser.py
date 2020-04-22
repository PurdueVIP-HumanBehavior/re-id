import os
import json
import galleries
import bbox_trigger
import constants

def parse_json_filename(filename):
    if not os.path.exists(filename):
        raise ValueError("json filename does not exist")
    with open(filename, "r") as jfile:
        json_cont = jfile.read()
    return json.loads(json_cont)

def extract_line_trigger_list(json_dict):
    lines = list()
    for vidname in json_dict:
        for trig in json_dict[vidname]:
            line = json_dict[vidname][trig]["line"]
            point = json_dict[vidname][trig]["point"]
            lines.append(bbox_trigger.VectorTrigger(vidname, line, point, constants.LT_MAX_DISTANCE, constants.LT_FRAME_OFFSET))
    return lines
