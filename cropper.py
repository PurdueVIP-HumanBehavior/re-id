def crop_image(img, box):
    # import ipdb; ipdb.set_trace()
    x = box[0][0]
    w = box[1][0] - box[0][0]
    y = box[0][1]
    h = box[1][1] - box[0][1]
    cropped = img[int(y) : int(y + h), int(x) : int(x + w)]
    return cropped
