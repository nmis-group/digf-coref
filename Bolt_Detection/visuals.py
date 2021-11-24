from utils import *

def show_prediction_two_plots(img, out, targets):
    img = img.cpu().permute(1,2,0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img2 = img.copy()
    boxes = np.array(targets['bboxes'])
    boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
    for b in boxes:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (100, 255, 200), 10)

            ((text_width, text_height), _) = cv2.getTextSize('Bolt', cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    

            #cv2.rectangle(img, (b[0], b[1] - int(1.3 * text_height)), (b[0] + text_width, b[1]), (255, 255, 255), -1)

            cv2.putText(
                img,
                text='Bolt',
                org=(int(b[0]), int(b[1]) - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, 
                color=(255, 255, 255), 
                thickness = 8,
                lineType=cv2.LINE_AA,
            )
    for b, a in zip(out[0][0], out[2][0]):
            cv2.rectangle(img2, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (80, 255, 0), 10)

            ((text_width, text_height), _) = cv2.getTextSize("{:.2f}".format(a), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    

            #cv2.rectangle(img, (b[0], b[1] - int(1.3 * text_height)), (b[0] + text_width, b[1]), (255, 255, 255), -1)

            cv2.putText(
                img2,
                text="{:.2f}".format(a),
                org=(int(b[0]), int(b[1]) - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, 
                color=(80, 255, 0), 
                thickness = 8,
                lineType=cv2.LINE_AA,
            )
    counted_bolts = len(out[2][0])
    plt.figure(figsize=(16,10))
    plt.subplot(1,2,1)
    plt.title('Groud Truth')
    plt.axis('off')
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.title('Prediction: Counted Bolts = {}'.format(counted_bolts))
    plt.imshow(img2)
    plt.axis('off')
    plt.show()
    
    
def show_prediction_one_plot(img, out, targets):
    img = img.cpu().permute(1,2,0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    #img2 = img.copy()
    boxes = np.array(targets['bboxes'])
    boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
    for b in boxes:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (100, 255, 200), 10)

            ((text_width, text_height), _) = cv2.getTextSize('Bolt', cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    

            #cv2.rectangle(img, (b[0], b[1] - int(1.3 * text_height)), (b[0] + text_width, b[1]), (255, 255, 255), -1)

            cv2.putText(
                img,
                text='Bolt',
                org=(int(b[0]), int(b[1]) - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, 
                color=(255, 255, 255), 
                thickness = 8,
                lineType=cv2.LINE_AA,
            )
    for b, a in zip(out[0][0], out[2][0]):
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (80, 255, 0), 10)

            ((text_width, text_height), _) = cv2.getTextSize("{:.2f}".format(a), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    

            #cv2.rectangle(img, (b[0], b[1] - int(1.3 * text_height)), (b[0] + text_width, b[1]), (255, 255, 255), -1)

            cv2.putText(
                img,
                text="{:.2f}".format(a),
                org=(int(b[0]), int(b[1]) - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, 
                color=(80, 255, 0), 
                thickness = 8,
                lineType=cv2.LINE_AA,
            )
    counted_bolts = len(out[2][0])
    plt.figure(figsize=(16,10))
    #plt.subplot(1,2,1)
    plt.title('Counted Bolts = {}'.format(counted_bolts))
    plt.axis('off')
    plt.imshow(img)
   # plt.subplot(1,2,2)
    #plt.title('Prediction')
    #plt.imshow(img2)
    #plt.axis('off')
    plt.show()


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