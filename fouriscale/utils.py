import math

def read_layer_settings(path):
    print(f"Reading FouriScale layer settings")
    layer_settings = []
    with open(path, 'r') as f:
        raw_lines = f.readlines()
        for raw_line in raw_lines:
            layer_settings.append(raw_line.rstrip('\n'))
    return layer_settings


def read_base_settings(path):
    print(f"Reading FouriScale base settings")
    base_settings = dict()
    with open(path, 'r') as f:
        raw_lines = f.readlines()
        for raw_line in raw_lines:
            aspect_ratio, dilate = raw_line.split(':')
            base_settings[aspect_ratio] = [float(s) for s in dilate.split(',')]
    return base_settings


def find_smallest_padding_pair(height, width, base_settings):
    # Initialize the minimum padding size to a large number and the result pair
    min_padding_size = float('inf')
    result_pair = None
    result_scale = None
    
    for aspect_ratio, size in base_settings.items():
        base_height, base_width = size
        scale_height = math.ceil(height / base_height)
        scale_width = math.ceil(width / base_width)

        if scale_height == scale_width:
            padding_height = base_height * scale_height
            padding_width = base_width * scale_width
            padding_size = (padding_height - height) + (padding_width - width)
            
            if padding_size < min_padding_size and padding_height >= height and padding_width >= width:
                min_padding_size = padding_size
                result_pair = (base_height, base_width)
                result_scale = aspect_ratio
                
    return result_pair, result_scale