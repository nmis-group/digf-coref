from utils import *

def collate_fn(batch):
    return tuple(zip(*batch))

class BoltDataset(Dataset):

    def __init__(self, df, transform, split):
        super().__init__()
        self.split = split
        self.label = 'Bolt'
        if split == 'test':
            df = df[df['fold']==4]
            df = df.reset_index(drop=True)
        if split == 'valid':
            df = df[df['fold']==3] 
            df = df.reset_index(drop=True)
        if split == 'train':
            df = df[df['fold'].isin([0,1,2])]
            df = df.reset_index(drop=True)
        self.df = df
        print(len(self.df))
        self.transform = transform

    def __getitem__(self, index: int):
        
        path = self.df.filename[index]
        #print(path)
        img= cv2.imread(path)        
        img = img.transpose(1,0,2)
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = self.df.bboxes[index]
        
        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes, class_labels=['Bolt']*len(bboxes))
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']

        # there is only one class
        labels = torch.ones(len(bboxes), dtype=torch.int64)

        return transformed_image, labels, torch.Tensor(transformed_bboxes)

    def __len__(self) -> int:
        return len(self.df)
    
    def show_data(self, index: int):
        """Visualizes a single bounding box on the image"""
        bboxes = self.df.bboxes[index]

        img = cv2.imread(self.df.filename[index])
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
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        