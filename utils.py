def crop_image(img, box):
    x = box[0][0]
    w = box[1][0] - box[0][0]
    y = box[0][1]
    h = box[1][1] - box[0][1]
    cropped = img[int(y):int(y + h), int(x):int(x + w)]
    return cropped


# TODO: (nhendy) this is not normalized?
def unitdotprod(vec1, vec2):
    return np.dot(vec1, np.transpose(vec2))
