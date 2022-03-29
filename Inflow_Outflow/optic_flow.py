



class optic_flow:

  def __init__(self):
    pass



def optic_flow(frame, old_frame, mask, p0):
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    # Default img initialization for return statement
    img = frame
    
    # If the mask is not empty
    if not (np.array_equal(np.empty(mask.shape), mask)):
        
        frame_gray = cv2.cvtColor(frame,
                              cv2.COLOR_BGR2GRAY)
        old_gray = cv2.cvtColor(old_frame,
                              cv2.COLOR_BGR2GRAY)
        
        if len(p0) <= 0:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None,
                            **feature_params)
    
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                            frame_gray,
                                            p0, None,
                                            **lk_params)
        
        # Select good points
        try:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        except Exception as e:
            pass
        
        
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, 
                                        good_old)):
            a, b = new.ravel()
            f, d = old.ravel()
           
            draw_mask = cv2.line(np.zeros_like(old_frame), (int(a), int(b)), (int(f), int(d)),
                            color[i].tolist(), 2)
            
            frame = cv2.circle(frame, (int(a), int(b)), 5,
                            color[i].tolist(), -1)
        # pdb.set_trace()
        img = cv2.add(frame, draw_mask)

    return img.astype(np.uint8), p0


