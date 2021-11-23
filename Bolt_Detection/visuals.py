from utils import *


def visualize_bbox_with_df(df, index):
    bboxes = df.bboxes[index]
    
    img = cv2.imread(df.filename[index])
    img = img.transpose(1,0,2)
    #img = np.fliplr(img)
    img = np.flipud(img)
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for b in bboxes:
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (100, 255, 200), 10)

        ((text_width, text_height), _) = cv2.getTextSize('Bolt', cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    

        #cv2.rectangle(img, (b[0], b[1] - int(1.3 * text_height)), (b[0] + text_width, b[1]), (255, 255, 255), -1)

        cv2.putText(
            img,
            text='Bolt',
            org=(b[0], b[1] - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3, 
            color=(255, 255, 255), 
            thickness = 8,
            lineType=cv2.LINE_AA,
        )
    return img
        
    
def visualize_bbox(img, bboxes):
    """Visualizes bounding boxes on the image"""
    for i in range(len(bboxes)):
        x_min, y_min, x_max, y_max = bboxes[i]
        
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (100, 255, 200), 10)

        ((text_width, text_height), _) = cv2.getTextSize('Bolt', cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    

        cv2.putText(
                img,
                text='Bolt',
                org=(int(x_min), int(y_min) - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3, 
                color=(255, 255, 255), 
                thickness = 8,
                lineType=cv2.LINE_AA,
        )
    return img