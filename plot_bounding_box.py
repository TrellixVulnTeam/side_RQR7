def make_bounding_box(img, anchors, colors, thickness=2):
    '''
    img -> (np.array) image array
    anchors -> (list) list of anchor(bounding_box, score, label)
    colors -> (list) list of colors for each label
    thickness -> (int) thickness of font and bounding box
    '''
    img1 = np.array(img).copy()
    bbs = list(map(lambda x: x[0], anchors))
    scores = list(map(lambda x: x[1], anchors))
    labels = list(map(lambda x: x[2], anchors))
    for i, bb in enumerate(bbs):
        # pdb.set_trace()
        img = cv2.rectangle(img1, tuple(bb[:2]), tuple(bb[2:]), colors[labels[i]], thickness)
        cv2.putText(img1, '{:.3f}'.format(scores[i]), (bb[0], bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[labels[i]], 2)
    return img1